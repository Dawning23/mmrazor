# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Quaternion and pose encoding utility functions."""

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Affine inverse
# ---------------------------------------------------------------------------

def affine_inverse(A: torch.Tensor) -> torch.Tensor:
    """Invert a batch of 4x4 (or 3x4) affine matrices efficiently via R^T."""
    R = A[..., :3, :3]   # ..., 3, 3
    T = A[..., :3, 3:]   # ..., 3, 1
    P = A[..., 3:, :]    # ..., 1, 4
    return torch.cat([torch.cat([R.mT, -R.mT @ T], dim=-1), P], dim=-2)


# ---------------------------------------------------------------------------
# Pose encoding  (matches original depth_anything_3.model.utils.transform)
# ---------------------------------------------------------------------------

def extri_intri_to_pose_encoding(extrinsics, intrinsics, image_size_hw=None):
    """Convert camera extrinsics and intrinsics to a compact pose encoding.

    Args:
        extrinsics: (B, S, 3, 4)  – c2w
        intrinsics: (B, S, 3, 3)
        image_size_hw: (H, W)

    Returns:
        pose_encoding: (B, S, 9)  – [T(3), quat(4), fov_h(1), fov_w(1)]
    """
    R = extrinsics[:, :, :3, :3]
    T = extrinsics[:, :, :3, 3]

    quat = mat_to_quat(R)
    H, W = image_size_hw
    fov_h = 2 * torch.atan((H / 2) / intrinsics[..., 1, 1])
    fov_w = 2 * torch.atan((W / 2) / intrinsics[..., 0, 0])
    pose_encoding = torch.cat([T, quat, fov_h[..., None], fov_w[..., None]], dim=-1).float()
    return pose_encoding


def pose_encoding_to_extri_intri(pose_encoding, image_size_hw=None):
    """Convert a pose encoding back to camera extrinsics and intrinsics.

    Args:
        pose_encoding: (B, S, 9)
        image_size_hw: (H, W)

    Returns:
        extrinsics: (B, S, 3, 4)
        intrinsics: (B, S, 3, 3)
    """
    T = pose_encoding[..., :3]
    quat = pose_encoding[..., 3:7]
    fov_h = pose_encoding[..., 7]
    fov_w = pose_encoding[..., 8]

    R = quat_to_mat(quat)
    extrinsics = torch.cat([R, T[..., None]], dim=-1)

    H, W = image_size_hw
    fy = (H / 2.0) / torch.clamp(torch.tan(fov_h / 2.0), 1e-6)
    fx = (W / 2.0) / torch.clamp(torch.tan(fov_w / 2.0), 1e-6)
    intrinsics = torch.zeros(pose_encoding.shape[:2] + (3, 3), device=pose_encoding.device)
    intrinsics[..., 0, 0] = fx
    intrinsics[..., 1, 1] = fy
    intrinsics[..., 0, 2] = W / 2
    intrinsics[..., 1, 2] = H / 2
    intrinsics[..., 2, 2] = 1.0

    return extrinsics, intrinsics


# ---------------------------------------------------------------------------
# Quaternion ↔ rotation matrix  (XYZW / ijkr, scalar-last)
# ---------------------------------------------------------------------------

def quat_to_mat(quaternions: torch.Tensor) -> torch.Tensor:
    """Convert quaternions (xyzw) to rotation matrices.

    Args:
        quaternions: (..., 4) quaternion in xyzw format
    Returns:
        (..., 3, 3) rotation matrix
    """
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
    """Convert rotation matrices to quaternions (xyzw).

    Args:
        matrix: (..., 3, 3) rotation matrix
    Returns:
        (..., 4) quaternion in xyzw format
    """
    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix shape {matrix.shape}.")

    batch_dim = matrix.shape[:-2]
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = torch.unbind(
        matrix.reshape(batch_dim + (9,)), dim=-1
    )

    q_abs = _sqrt_positive_part(
        torch.stack(
            [
                1.0 + m00 + m11 + m22,
                1.0 + m00 - m11 - m22,
                1.0 - m00 + m11 - m22,
                1.0 - m00 - m11 + m22,
            ],
            dim=-1,
        )
    )

    quat_by_rijk = torch.stack(
        [
            torch.stack([q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01], dim=-1),
            torch.stack([m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20], dim=-1),
            torch.stack([m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21], dim=-1),
            torch.stack([m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2], dim=-1),
        ],
        dim=-2,
    )

    flr = torch.tensor(0.1).to(dtype=q_abs.dtype, device=q_abs.device)
    quat_candidates = quat_by_rijk / (2.0 * q_abs[..., None].max(flr))

    out = quat_candidates[
        F.one_hot(q_abs.argmax(dim=-1), num_classes=4) > 0.5, :
    ].reshape(batch_dim + (4,))

    # Convert from rijk to ijkr
    out = out[..., [1, 2, 3, 0]]
    out = standardize_quaternion(out)
    return out


def standardize_quaternion(quaternions: torch.Tensor) -> torch.Tensor:
    """Standardize quaternion to have non-negative real part."""
    return torch.where(quaternions[..., 3:4] < 0, -quaternions, quaternions)


def _sqrt_positive_part(x: torch.Tensor) -> torch.Tensor:
    ret = torch.zeros_like(x)
    positive_mask = x > 0
    if torch.is_grad_enabled():
        ret[positive_mask] = torch.sqrt(x[positive_mask])
    else:
        ret = torch.where(positive_mask, torch.sqrt(x), ret)
    return ret
