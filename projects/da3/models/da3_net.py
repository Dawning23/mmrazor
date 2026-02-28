# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Depth-Anything-3 top-level network modules (ported to mmrazor)."""

import logging
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from einops import rearrange

from .utils.alignment import (
    apply_metric_scaling,
    compute_alignment_mask,
    compute_sky_mask,
    least_squares_scale_scalar,
    sample_tensor_for_quantile,
    set_sky_regions_to_max_depth,
)
from .utils.ray_utils import get_extrinsic_from_camray
from .utils.transform import affine_inverse, pose_encoding_to_extri_intri

logger = logging.getLogger("da3")


class DepthAnything3Net(nn.Module):
    """
    Depth-Anything-3 network.

    Args:
        net: DinoV2 backbone module
        head: DPT or DualDPT head module
        cam_enc: Optional CameraEnc module
        cam_dec: Optional CameraDec module
    """

    def __init__(
        self,
        net: nn.Module,
        head: nn.Module,
        cam_enc: nn.Module = None,
        cam_dec: nn.Module = None,
    ):
        super().__init__()
        self.net = net
        self.head = head
        self.cam_enc = cam_enc
        self.cam_dec = cam_dec

        # Backbone's get_intermediate_layers already strips cls+register tokens,
        # so the head should start at index 0.
        self.patch_start_idx = 0
        self.patch_size = getattr(
            getattr(self.net, 'pretrained', self.net), 'patch_size', 14
        )

    def forward(
        self,
        image: torch.Tensor,
        extrinsics: Optional[torch.Tensor] = None,
        intrinsics: Optional[torch.Tensor] = None,
        export_feat_layers: Optional[List[int]] = None,
        infer_gs: bool = False,
        use_ray_pose: bool = False,
        ref_view_strategy: str = "saddle_balanced",
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            image: (B, S, 3, H, W)
            extrinsics: Optional (B, S, 4, 4)
            intrinsics: Optional (B, S, 3, 3)
            export_feat_layers: layer indices for feature export
            infer_gs: not supported in this port
            use_ray_pose: use ray-based pose estimation
            ref_view_strategy: reference view selection strategy

        Returns:
            Dict with keys: depth, depth_conf, sky, extrinsics, etc.
        """
        B, S, C, H, W = image.shape

        # Prepare camera tokens if extrinsics provided and cam_enc exists
        cam_token = None
        if extrinsics is not None and self.cam_enc is not None:
            with torch.autocast(device_type=image.device.type, enabled=False):
                cam_token = self.cam_enc(extrinsics, intrinsics, image.shape[-2:])

        # Run backbone
        backbone_kwargs = dict(ref_view_strategy=ref_view_strategy)
        if cam_token is not None:
            backbone_kwargs["cam_token"] = cam_token
        if export_feat_layers is None:
            export_feat_layers = []

        feats, aux_feats = self.net(
            image,
            export_feat_layers=export_feat_layers,
            **backbone_kwargs,
        )

        # Run depth head
        out = self._process_depth_head(feats, H, W)

        # Camera pose estimation
        out = self._process_camera_estimation(
            out, feats, extrinsics, use_ray_pose, H, W
        )

        # Add auxiliary features if requested
        if aux_feats:
            out["aux_feats"] = aux_feats

        return out

    def _process_depth_head(
        self, feats, H: int, W: int
    ) -> Dict[str, torch.Tensor]:
        """Process features through depth head."""
        head_out = self.head(
            feats, H, W,
            patch_start_idx=self.patch_start_idx,
        )
        return dict(head_out)

    def _process_camera_estimation(
        self,
        out: Dict[str, torch.Tensor],
        feats,
        extrinsics: Optional[torch.Tensor],
        use_ray_pose: bool,
        H: int,
        W: int,
    ) -> Dict[str, torch.Tensor]:
        """Process camera pose estimation."""
        # Camera decoder path
        if self.cam_dec is not None and not use_ray_pose:
            with torch.autocast(device_type=feats[0][0].device.type, enabled=False):
                pose_enc = self.cam_dec(feats[-1][1])
                # Remove ray info
                if "ray" in out:
                    del out["ray"]
                if "ray_conf" in out:
                    del out["ray_conf"]
                # Convert pose encoding to extrinsics and intrinsics
                c2w, ixt = pose_encoding_to_extri_intri(pose_enc, (H, W))
                out["extrinsics"] = affine_inverse(c2w)
                out["intrinsics"] = ixt

        # Ray-based pose estimation path
        if use_ray_pose and "ray" in out:
            ph = H // self.patch_size
            pw = W // self.patch_size
            ray = out["ray"]
            ray_conf = out.get("ray_conf", None)
            pred_ext, pred_fl, pred_pp = get_extrinsic_from_camray(
                ray, ray_conf, ph, pw
            )
            out["extrinsics"] = pred_ext
            out["focal_lengths"] = pred_fl
            out["principal_points"] = pred_pp

        return out


class NestedDepthAnything3Net(nn.Module):
    """
    Nested Depth-Anything-3 network for metric depth estimation.

    Combines an anyview (multi-view) model and a metric (single-view) model
    with metric scaling alignment.

    Args:
        anyview: DepthAnything3Net for multi-view depth estimation
        metric: DepthAnything3Net for metric depth estimation
    """

    def __init__(
        self,
        anyview: DepthAnything3Net,
        metric: DepthAnything3Net,
    ):
        super().__init__()
        self.anyview = anyview
        self.metric = metric

    def forward(
        self,
        image: torch.Tensor,
        extrinsics: Optional[torch.Tensor] = None,
        intrinsics: Optional[torch.Tensor] = None,
        export_feat_layers: Optional[List[int]] = None,
        infer_gs: bool = False,
        use_ray_pose: bool = False,
        ref_view_strategy: str = "saddle_balanced",
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for nested model.

        Args: same as DepthAnything3Net.forward
        Returns: Dict with metric-aligned depth and camera parameters
        """
        # Run anyview model
        anyview_out = self.anyview(
            image, extrinsics, intrinsics,
            export_feat_layers, infer_gs, use_ray_pose,
            ref_view_strategy,
        )

        # Run metric model (single-view, no camera info)
        metric_out = self.metric(image)

        # Align and merge
        out = self._align_metric(anyview_out, metric_out, intrinsics)
        return out

    def _align_metric(
        self,
        anyview_out: Dict[str, torch.Tensor],
        metric_out: Dict[str, torch.Tensor],
        intrinsics: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Align anyview depth to metric scale."""
        out = dict(anyview_out)

        depth = anyview_out["depth"]
        depth_conf = anyview_out.get("depth_conf", None)
        metric_depth = metric_out["depth"]
        sky = anyview_out.get("sky", None)

        # Apply metric scaling if intrinsics available
        if intrinsics is not None:
            metric_depth = apply_metric_scaling(metric_depth, intrinsics)

        # Compute sky mask
        non_sky_mask = torch.ones_like(depth, dtype=torch.bool)
        if sky is not None:
            non_sky_mask = compute_sky_mask(sky)

        # Per-view alignment
        B, S = depth.shape[:2]
        aligned_depth = depth.clone()

        for b in range(B):
            for s in range(S):
                d = depth[b, s]
                md = metric_depth[b, s]
                dc = depth_conf[b, s] if depth_conf is not None else torch.ones_like(d)
                nsm = non_sky_mask[b, s]

                # Compute median confidence for thresholding
                dc_flat = sample_tensor_for_quantile(dc[nsm])
                if dc_flat.numel() > 0:
                    median_conf = dc_flat.quantile(0.5)
                else:
                    median_conf = torch.tensor(0.0, device=d.device)

                mask = compute_alignment_mask(dc, nsm, d, md, median_conf)

                if mask.sum() > 10:
                    scale = least_squares_scale_scalar(md[mask], d[mask])
                    aligned_depth[b, s] = d * scale

        # Set sky to max depth
        if sky is not None:
            aligned_depth, depth_conf = set_sky_regions_to_max_depth(
                aligned_depth, depth_conf, non_sky_mask
            )

        out["depth"] = aligned_depth
        if depth_conf is not None:
            out["depth_conf"] = depth_conf
        out["metric_depth"] = metric_depth

        return out
