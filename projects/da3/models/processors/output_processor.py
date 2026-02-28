# Copyright (c) OpenMMLab. All rights reserved.
"""DA3 Output Processor — mmlab-style postprocessing for Depth-Anything-3.

Converts raw model output tensors to structured numpy results,
matching the official OutputProcessor pipeline.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np
import torch

try:
    from mmrazor.registry import TASK_UTILS
    HAS_REGISTRY = True
except ImportError:
    HAS_REGISTRY = False


@dataclass
class DA3Prediction:
    """Structured prediction output from DA3 inference."""
    depth: np.ndarray                          # (N, H, W) float32
    conf: Optional[np.ndarray] = None          # (N, H, W) float32
    extrinsics: Optional[np.ndarray] = None    # (N, 3, 4) float32
    intrinsics: Optional[np.ndarray] = None    # (N, 3, 3) float32
    processed_images: Optional[np.ndarray] = None  # (N, H, W, 3) uint8
    sky: Optional[np.ndarray] = None           # (N, H, W) bool
    is_metric: int = 0


class DA3OutputProcessor:
    """Converts raw model output dict to DA3Prediction.

    Handles tensor→numpy conversion and batch dimension removal.
    Compatible with the official depth_anything_3 OutputProcessor.
    """

    def __init__(self):
        pass

    def __call__(self, model_output: Dict[str, torch.Tensor]) -> DA3Prediction:
        """Convert raw model output to DA3Prediction.

        Args:
            model_output: Dict with keys like ``depth``, ``depth_conf``,
                ``extrinsics``, ``intrinsics``. Shapes: ``(B, S, ...)``.

        Returns:
            DA3Prediction with batch dim squeezed.
        """
        depth = model_output["depth"].squeeze(0).squeeze(-1).cpu().numpy()

        conf = model_output.get("depth_conf", None)
        if conf is not None:
            conf = conf.squeeze(0).cpu().numpy()

        extrinsics = model_output.get("extrinsics", None)
        if extrinsics is not None:
            extrinsics = extrinsics.squeeze(0).cpu().numpy()

        intrinsics = model_output.get("intrinsics", None)
        if intrinsics is not None:
            intrinsics = intrinsics.squeeze(0).cpu().numpy()

        sky = model_output.get("sky", None)
        if sky is not None:
            sky = sky.squeeze(0).cpu().numpy() >= 0.5

        return DA3Prediction(
            depth=depth,
            conf=conf,
            extrinsics=extrinsics,
            intrinsics=intrinsics,
            sky=sky,
        )


if HAS_REGISTRY:
    TASK_UTILS.register_module()(DA3OutputProcessor)
