# Copyright (c) OpenMMLab. All rights reserved.
"""DA3 Inferencer — high-level inference class for Depth-Anything-3.

Usage with config:
    inferencer = DA3Inferencer(cfg)
    prediction = inferencer.inference(images)
"""

from typing import Dict, List, Optional, Union

import numpy as np
import torch
import torch.nn as nn

from .processors.input_processor import DA3InputProcessor
from .processors.output_processor import DA3OutputProcessor, DA3Prediction
from .utils.transform import affine_inverse


class DA3Inferencer:
    """High-level DA3 inference pipeline.

    Wraps model + input/output processors into a single callable that
    mirrors the official ``DepthAnything3.inference()`` API.

    Args:
        model (nn.Module): DA3 network (DepthAnything3Net).
        input_processor (DA3InputProcessor): Preprocessing pipeline.
        output_processor (DA3OutputProcessor): Postprocessing pipeline.
        device (str): Device for inference. Default: ``cuda``.
    """

    def __init__(
        self,
        model: nn.Module,
        input_processor: Optional[DA3InputProcessor] = None,
        output_processor: Optional[DA3OutputProcessor] = None,
        device: str = 'cuda',
    ):
        self.model = model.eval()
        self.input_processor = input_processor or DA3InputProcessor()
        self.output_processor = output_processor or DA3OutputProcessor()
        self.device = torch.device(device)
        self.model.to(self.device)

    def inference(
        self,
        images: List[Union[np.ndarray, str]],
        extrinsics: Optional[np.ndarray] = None,
        intrinsics: Optional[np.ndarray] = None,
    ) -> DA3Prediction:
        """Run full inference pipeline on a list of images.

        Args:
            images: List of image paths or numpy arrays.
            extrinsics: Optional (N, 4, 4) camera extrinsics.
            intrinsics: Optional (N, 3, 3) camera intrinsics.

        Returns:
            DA3Prediction with depth, conf, extrinsics, intrinsics, processed_images.
        """
        # 1. Preprocess
        imgs_cpu, ext_t, ixt_t = self.input_processor(images, extrinsics, intrinsics)

        # 2. Prepare model inputs: add batch dim
        imgs = imgs_cpu.unsqueeze(0).float().to(self.device)
        ex_t = ext_t.unsqueeze(0).float().to(self.device) if ext_t is not None else None
        in_t = ixt_t.unsqueeze(0).float().to(self.device) if ixt_t is not None else None

        # 3. Normalize extrinsics
        ex_t = self._normalize_extrinsics(ex_t)

        # 4. Forward (fp32)
        with torch.no_grad():
            raw_output = self.model(imgs, ex_t, in_t)

        # 5. Postprocess
        prediction = self.output_processor(raw_output)

        # 6. Add processed images
        prediction.processed_images = DA3InputProcessor.denormalize(imgs_cpu)

        return prediction

    def _normalize_extrinsics(self, ex_t: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        """Normalize extrinsics (same as official API)."""
        if ex_t is None:
            return None
        transform = affine_inverse(ex_t[:, :1])
        ex_t = ex_t @ transform
        c2ws = affine_inverse(ex_t)
        translations = c2ws[..., :3, 3]
        dists = translations.norm(dim=-1)
        median_dist = torch.clamp(torch.median(dists), min=1e-1)
        ex_t[..., :3, 3] = ex_t[..., :3, 3] / median_dist
        return ex_t

    @staticmethod
    def from_config(cfg: dict) -> 'DA3Inferencer':
        """Build DA3Inferencer from a config dict.

        Args:
            cfg: Config dict with keys ``model``, ``input_processor``,
                ``output_processor``, ``device``, ``weights``.

        Returns:
            DA3Inferencer instance.
        """
        from .dinov2 import DinoV2
        from .heads.dpt import DPT
        from .heads.dualdpt import DualDPT
        from .camera.cam_enc import CameraEnc
        from .camera.cam_dec import CameraDec
        from .da3_net import DepthAnything3Net

        # Build model from config
        model_cfg = cfg['model']
        variant = model_cfg['variant']

        if variant == 'da3-large':
            net = DinoV2(**model_cfg.get('backbone', dict(
                name='vitl', out_layers=[11, 15, 19, 23],
                alt_start=8, qknorm_start=8, rope_start=8, cat_token=True,
            )))
            head = DualDPT(**model_cfg.get('head', dict(
                dim_in=2048, output_dim=2, features=256,
                out_channels=[256, 512, 1024, 1024],
            )))
            cam_enc = CameraEnc(**model_cfg.get('cam_enc', dict(dim_out=1024)))
            cam_dec = CameraDec(**model_cfg.get('cam_dec', dict(dim_in=2048)))
            model = DepthAnything3Net(net=net, head=head, cam_enc=cam_enc, cam_dec=cam_dec)

        elif variant == 'da3-small':
            net = DinoV2(**model_cfg.get('backbone', dict(
                name='vits', out_layers=[5, 7, 9, 11],
                alt_start=4, qknorm_start=4, rope_start=4, cat_token=True,
            )))
            head = DualDPT(**model_cfg.get('head', dict(
                dim_in=768, output_dim=2, features=256,
                out_channels=[256, 512, 1024, 1024],
            )))
            cam_enc = CameraEnc(**model_cfg.get('cam_enc', dict(dim_out=384)))
            cam_dec = CameraDec(**model_cfg.get('cam_dec', dict(dim_in=768)))
            model = DepthAnything3Net(net=net, head=head, cam_enc=cam_enc, cam_dec=cam_dec)

        else:
            raise ValueError(f"Unknown variant: {variant}")

        # Load weights
        weights_path = cfg.get('weights', None)
        if weights_path is not None:
            state_dict = torch.load(weights_path, map_location='cpu', weights_only=True)
            # Remap backbone. -> net.
            state_dict = {
                k.replace('backbone.', 'net.', 1) if k.startswith('backbone.') else k: v
                for k, v in state_dict.items()
            }
            model.load_state_dict(state_dict, strict=True)

        # Build processors
        input_cfg = cfg.get('input_processor', {})
        output_cfg = cfg.get('output_processor', {})
        input_proc = DA3InputProcessor(**input_cfg)
        output_proc = DA3OutputProcessor(**output_cfg)

        device = cfg.get('device', 'cuda')
        return DA3Inferencer(model, input_proc, output_proc, device)
