# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Camera Encoder: encodes camera extrinsics/intrinsics into tokens.

Matches the original depth_anything_3.model.cam_enc architecture exactly.
"""

import torch.nn as nn

from ..utils.attention import Mlp
from ..utils.block import Block
from ..utils.transform import affine_inverse, extri_intri_to_pose_encoding


class CameraEnc(nn.Module):
    """
    CameraHead predicts camera parameters from token representations using
    iterative refinement.

    It applies a series of transformer blocks (the "trunk") to dedicated
    camera tokens.
    """

    def __init__(
        self,
        dim_out: int = 1024,
        dim_in: int = 9,
        trunk_depth: int = 4,
        target_dim: int = 9,
        num_heads: int = 16,
        mlp_ratio: int = 4,
        init_values: float = 0.01,
        **kwargs,
    ):
        super().__init__()
        self.target_dim = target_dim
        self.trunk_depth = trunk_depth
        self.trunk = nn.Sequential(
            *[
                Block(
                    dim=dim_out,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=True,
                    init_values=init_values,
                )
                for _ in range(trunk_depth)
            ]
        )
        self.token_norm = nn.LayerNorm(dim_out)
        self.trunk_norm = nn.LayerNorm(dim_out)
        self.pose_branch = Mlp(
            in_features=dim_in,
            hidden_features=dim_out // 2,
            out_features=dim_out,
            drop=0,
        )

    def forward(self, ext, ixt, image_size) -> tuple:
        c2ws = affine_inverse(ext)
        pose_encoding = extri_intri_to_pose_encoding(
            c2ws,
            ixt,
            image_size,
        )
        pose_tokens = self.pose_branch(pose_encoding)
        pose_tokens = self.token_norm(pose_tokens)
        pose_tokens = self.trunk(pose_tokens)
        pose_tokens = self.trunk_norm(pose_tokens)
        return pose_tokens
