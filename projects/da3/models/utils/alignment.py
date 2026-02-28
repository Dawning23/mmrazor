# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Alignment utilities for depth estimation and metric scaling."""

from typing import Tuple
import torch


def least_squares_scale_scalar(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Compute least squares scale factor s such that a ~ s * b."""
    if a.shape != b.shape:
        raise ValueError(f"Shape mismatch: {a.shape} vs {b.shape}")
    num = torch.dot(a.reshape(-1), b.reshape(-1))
    den = torch.dot(b.reshape(-1), b.reshape(-1)).clamp_min(eps)
    return num / den


def compute_sky_mask(sky_prediction: torch.Tensor, threshold: float = 0.3) -> torch.Tensor:
    """Compute non-sky mask from sky prediction."""
    return sky_prediction < threshold


def compute_alignment_mask(
    depth_conf, non_sky_mask, depth, metric_depth, median_conf,
    min_depth_threshold=1e-3, min_metric_depth_threshold=1e-2,
):
    return (
        (depth_conf >= median_conf)
        & non_sky_mask
        & (metric_depth > min_metric_depth_threshold)
        & (depth > min_depth_threshold)
    )


def sample_tensor_for_quantile(tensor, max_samples=100000):
    if tensor.numel() <= max_samples:
        return tensor
    idx = torch.randperm(tensor.numel(), device=tensor.device)[:max_samples]
    return tensor.flatten()[idx]


def apply_metric_scaling(depth, intrinsics, scale_factor=300.0):
    focal_length = (intrinsics[:, :, 0, 0] + intrinsics[:, :, 1, 1]) / 2
    return depth * (focal_length[:, :, None, None] / scale_factor)


def set_sky_regions_to_max_depth(
    depth, depth_conf, non_sky_mask, max_depth=200.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    depth = depth.clone()
    depth[~non_sky_mask] = max_depth
    if depth_conf is not None:
        depth_conf = depth_conf.clone()
        depth_conf[~non_sky_mask] = 1.0
        return depth, depth_conf
    else:
        return depth, None
