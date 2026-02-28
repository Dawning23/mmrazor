# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Geometry utility functions."""

from types import SimpleNamespace
from typing import Optional
import numpy as np
import torch
import torch.nn.functional as F
from einops import einsum


def as_homogeneous(ext):
    """Accept (..., 3,4) or (..., 4,4) extrinsics, return (...,4,4) homogeneous matrix."""
    if isinstance(ext, torch.Tensor):
        if ext.shape[-2:] == (4, 4):
            return ext
        elif ext.shape[-2:] == (3, 4):
            ones = torch.zeros_like(ext[..., :1, :4])
            ones[..., 0, 3] = 1.0
            return torch.cat([ext, ones], dim=-2)
        else:
            raise ValueError(f"Invalid shape for torch.Tensor: {ext.shape}")
    elif isinstance(ext, np.ndarray):
        if ext.shape[-2:] == (4, 4):
            return ext
        elif ext.shape[-2:] == (3, 4):
            ones = np.zeros_like(ext[..., :1, :4])
            ones[..., 0, 3] = 1.0
            return np.concatenate([ext, ones], axis=-2)
        else:
            raise ValueError(f"Invalid shape for np.ndarray: {ext.shape}")
    else:
        raise TypeError("Input must be a torch.Tensor or np.ndarray.")


@torch.jit.script
def affine_inverse(A: torch.Tensor):
    R = A[..., :3, :3]
    T = A[..., :3, 3:]
    P = A[..., 3:, :]
    return torch.cat([torch.cat([R.mT, -R.mT @ T], dim=-1), P], dim=-2)


def quat_to_mat(quaternions: torch.Tensor) -> torch.Tensor:
    """Quaternion (XYZW) to rotation matrix."""
    i, j, k, r = torch.unbind(quaternions, -1)
    two_s = 2.0 / (quaternions * quaternions).sum(-1)
    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))


def mat_to_quat(matrix: torch.Tensor) -> torch.Tensor:
    """Rotation matrix to quaternion (XYZW)."""
    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix shape {matrix.shape}.")
    batch_dim = matrix.shape[:-2]
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = torch.unbind(
        matrix.reshape(batch_dim + (9,)), dim=-1
    )
    q_abs = _sqrt_positive_part(
        torch.stack([
            1.0 + m00 + m11 + m22,
            1.0 + m00 - m11 - m22,
            1.0 - m00 + m11 - m22,
            1.0 - m00 - m11 + m22,
        ], dim=-1)
    )
    quat_by_rijk = torch.stack([
        torch.stack([q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01], dim=-1),
        torch.stack([m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20], dim=-1),
        torch.stack([m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21], dim=-1),
        torch.stack([m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2], dim=-1),
    ], dim=-2)
    flr = torch.tensor(0.1).to(dtype=q_abs.dtype, device=q_abs.device)
    quat_candidates = quat_by_rijk / (2.0 * q_abs[..., None].max(flr))
    out = quat_candidates[F.one_hot(q_abs.argmax(dim=-1), num_classes=4) > 0.5, :].reshape(
        batch_dim + (4,)
    )
    out = out[..., [1, 2, 3, 0]]
    out = standardize_quaternion(out)
    return out


def _sqrt_positive_part(x: torch.Tensor) -> torch.Tensor:
    ret = torch.zeros_like(x)
    positive_mask = x > 0
    if torch.is_grad_enabled():
        ret[positive_mask] = torch.sqrt(x[positive_mask])
    else:
        ret = torch.where(positive_mask, torch.sqrt(x), ret)
    return ret


def standardize_quaternion(quaternions: torch.Tensor) -> torch.Tensor:
    return torch.where(quaternions[..., 3:4] < 0, -quaternions, quaternions)


def homogenize_points(points: torch.Tensor) -> torch.Tensor:
    return torch.cat([points, torch.ones_like(points[..., :1])], dim=-1)


def homogenize_vectors(vectors: torch.Tensor) -> torch.Tensor:
    return torch.cat([vectors, torch.zeros_like(vectors[..., :1])], dim=-1)


def transform_rigid(homogeneous_coordinates, transformation):
    return einsum(transformation, homogeneous_coordinates.to(transformation.dtype), "... i j, ... j -> ... i")


def transform_cam2world(homogeneous_coordinates, extrinsics):
    return transform_rigid(homogeneous_coordinates, extrinsics)


def inverse_intrinsic_matrix(ixts):
    return torch.inverse(ixts)


def normalize_homogenous_points(points):
    return points / points[..., -1:]


def pixel_space_to_camera_space(pixel_space_points, depth, intrinsics):
    pixel_space_points = homogenize_points(pixel_space_points)
    camera_space_points = torch.einsum(
        "b v i j , h w j -> b v h w i", inverse_intrinsic_matrix(intrinsics), pixel_space_points
    )
    camera_space_points = camera_space_points * depth
    return camera_space_points


def camera_space_to_world_space(camera_space_points, c2w):
    camera_space_points = homogenize_points(camera_space_points)
    world_space_points = torch.einsum("b v i j , b v h w j -> b v h w i", c2w, camera_space_points)
    return world_space_points[..., :3]


def unproject_depth(depth, intrinsics, c2w=None, ixt_normalized=False, num_patches_x=None, num_patches_y=None):
    if c2w is None:
        c2w = torch.eye(4, device=depth.device, dtype=depth.dtype)
        c2w = c2w[None, None].repeat(depth.shape[0], depth.shape[1], 1, 1)

    if not ixt_normalized:
        h, w = depth.shape[-3], depth.shape[-2]
        x_grid, y_grid = torch.meshgrid(
            torch.arange(w, device=depth.device, dtype=depth.dtype),
            torch.arange(h, device=depth.device, dtype=depth.dtype),
            indexing="xy",
        )
    else:
        assert num_patches_x is not None and num_patches_y is not None
        dx = 1 / num_patches_x
        dy = 1 / num_patches_y
        max_y = 1 - dy
        min_y = -max_y
        max_x = 1 - dx
        min_x = -max_x
        grid_shift = 1.0
        y_grid, x_grid = torch.meshgrid(
            torch.linspace(min_y + grid_shift, max_y + grid_shift, num_patches_y,
                           dtype=torch.float32, device=depth.device),
            torch.linspace(min_x + grid_shift, max_x + grid_shift, num_patches_x,
                           dtype=torch.float32, device=depth.device),
            indexing="ij",
        )

    pixel_space_points = torch.stack((x_grid, y_grid), dim=-1)
    camera_points = pixel_space_to_camera_space(pixel_space_points, depth, intrinsics)
    world_points = camera_space_to_world_space(camera_points, c2w)
    return world_points


def map_pdf_to_opacity(pdf, global_step=0, opacity_mapping=None):
    if opacity_mapping is not None:
        cfg = SimpleNamespace(**opacity_mapping)
        x = cfg.initial + min(global_step / cfg.warm_up, 1) * (cfg.final - cfg.initial)
    else:
        x = 0.0
    exponent = 2**x
    return 0.5 * (1 - (1 - pdf) ** exponent + pdf ** (1 / exponent))
