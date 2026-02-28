# Copyright (c) OpenMMLab. All rights reserved.
"""DA3 Input Processor — mmlab-style preprocessing for Depth-Anything-3.

Registered as a TRANSFORMS module, matching the official InputProcessor pipeline:
  1. Load image (path / numpy / PIL)
  2. Resize longest side to ``process_res`` while preserving aspect ratio
  3. Make dimensions divisible by ``patch_size`` (crop or resize)
  4. ImageNet normalize
  5. Stack into (N, 3, H, W) tensor
"""

from typing import List, Optional, Tuple, Union

import cv2
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image

try:
    from mmrazor.registry import TRANSFORMS
    HAS_REGISTRY = True
except ImportError:
    HAS_REGISTRY = False

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class DA3InputProcessor:
    """Preprocesses images for DA3 inference.

    Args:
        process_res (int): Target resolution (longest side). Default: 504.
        process_res_method (str): Resize method. Default: ``upper_bound_resize``.
        patch_size (int): Patch size for divisibility. Default: 14.
    """

    NORMALIZE = T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    PATCH_SIZE = 14

    def __init__(
        self,
        process_res: int = 504,
        process_res_method: str = 'upper_bound_resize',
        patch_size: int = 14,
    ):
        self.process_res = process_res
        self.process_res_method = process_res_method
        self.PATCH_SIZE = patch_size

    def __call__(
        self,
        images: List[Union[np.ndarray, Image.Image, str]],
        extrinsics: Optional[np.ndarray] = None,
        intrinsics: Optional[np.ndarray] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Preprocess a list of images.

        Args:
            images: List of images (paths, numpy BGR, or PIL RGB).
            extrinsics: Optional (N, 4, 4) camera extrinsics.
            intrinsics: Optional (N, 3, 3) camera intrinsics.

        Returns:
            imgs_tensor: (N, 3, H, W) float tensor, ImageNet-normalized.
            extrinsics_t: Optional (N, 4, 4) tensor.
            intrinsics_t: Optional (N, 3, 3) tensor.
        """
        processed = []
        ixt_list = []
        for i, img in enumerate(images):
            ixt = intrinsics[i] if intrinsics is not None else None
            tensor, ixt_out = self._process_one(img, ixt)
            processed.append(tensor)
            ixt_list.append(ixt_out)

        # Unify shapes: center-crop to smallest H, W
        processed, ixt_list = self._unify_shapes(processed, ixt_list)
        imgs_tensor = torch.stack(processed, dim=0)  # (N, 3, H, W)

        # Convert camera params to tensors
        ext_t = None
        if extrinsics is not None:
            ext_t = torch.from_numpy(extrinsics).float()
        ixt_t = None
        if intrinsics is not None:
            stacked = np.stack([x for x in ixt_list if x is not None])
            ixt_t = torch.from_numpy(stacked).float()

        return imgs_tensor, ext_t, ixt_t

    def _process_one(
        self, img: Union[np.ndarray, Image.Image, str],
        intrinsic: Optional[np.ndarray] = None,
    ) -> Tuple[torch.Tensor, Optional[np.ndarray]]:
        pil_img = self._load_image(img)
        orig_w, orig_h = pil_img.size

        # Resize
        pil_img = self._resize_image(pil_img, self.process_res, self.process_res_method)
        w, h = pil_img.size
        intrinsic = self._resize_ixt(intrinsic, orig_w, orig_h, w, h)

        # Make divisible by patch_size
        if self.process_res_method.endswith('resize'):
            pil_img = self._make_divisible_by_resize(pil_img, self.PATCH_SIZE)
        elif self.process_res_method.endswith('crop'):
            pil_img = self._make_divisible_by_crop(pil_img, self.PATCH_SIZE)
        new_w, new_h = pil_img.size
        intrinsic = self._resize_ixt(intrinsic, w, h, new_w, new_h)

        # Normalize
        tensor = T.ToTensor()(pil_img)
        tensor = self.NORMALIZE(tensor)
        return tensor, intrinsic

    # ── Image loading ──

    def _load_image(self, img: Union[np.ndarray, Image.Image, str]) -> Image.Image:
        if isinstance(img, str):
            return Image.open(img).convert('RGB')
        if isinstance(img, np.ndarray):
            if img.ndim == 3 and img.shape[-1] == 3:
                return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            return Image.fromarray(img)
        if isinstance(img, Image.Image):
            return img.convert('RGB')
        raise TypeError(f"Unsupported image type: {type(img)}")

    # ── Resize (cv2-based, matching official DA3) ──

    def _resize_image(self, img: Image.Image, target_size: int, method: str) -> Image.Image:
        if 'upper_bound' in method:
            return self._resize_longest_side(img, target_size)
        elif 'lower_bound' in method:
            return self._resize_shortest_side(img, target_size)
        return img

    def _resize_longest_side(self, img: Image.Image, target: int) -> Image.Image:
        w, h = img.size
        longest = max(w, h)
        if longest == target:
            return img
        scale = target / float(longest)
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))
        interpolation = cv2.INTER_CUBIC if scale > 1.0 else cv2.INTER_AREA
        arr = cv2.resize(np.asarray(img), (new_w, new_h), interpolation=interpolation)
        return Image.fromarray(arr)

    def _resize_shortest_side(self, img: Image.Image, target: int) -> Image.Image:
        w, h = img.size
        shortest = min(w, h)
        if shortest == target:
            return img
        scale = target / float(shortest)
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))
        interpolation = cv2.INTER_CUBIC if scale > 1.0 else cv2.INTER_AREA
        arr = cv2.resize(np.asarray(img), (new_w, new_h), interpolation=interpolation)
        return Image.fromarray(arr)

    # ── Patch divisibility (cv2-based, matching official DA3) ──

    def _make_divisible_by_crop(self, img: Image.Image, patch: int) -> Image.Image:
        w, h = img.size
        new_w = (w // patch) * patch
        new_h = (h // patch) * patch
        if new_w == w and new_h == h:
            return img
        left = (w - new_w) // 2
        top = (h - new_h) // 2
        return img.crop((left, top, left + new_w, top + new_h))

    def _make_divisible_by_resize(self, img: Image.Image, patch: int) -> Image.Image:
        w, h = img.size

        def nearest_multiple(x: int, p: int) -> int:
            down = (x // p) * p
            up = down + p
            return up if abs(up - x) <= abs(x - down) else down

        new_w = max(1, nearest_multiple(w, patch))
        new_h = max(1, nearest_multiple(h, patch))
        if new_w == w and new_h == h:
            return img
        upscale = (new_w > w) or (new_h > h)
        interpolation = cv2.INTER_CUBIC if upscale else cv2.INTER_AREA
        arr = cv2.resize(np.asarray(img), (new_w, new_h), interpolation=interpolation)
        return Image.fromarray(arr)

    # ── Intrinsics adjustment ──

    def _resize_ixt(self, ixt, orig_w, orig_h, new_w, new_h):
        if ixt is None:
            return None
        ixt = ixt.copy()
        ixt[0] *= new_w / orig_w
        ixt[1] *= new_h / orig_h
        return ixt

    # ── Shape unification ──

    def _unify_shapes(self, tensors, ixts):
        if len(tensors) <= 1:
            return tensors, ixts
        min_h = min(t.shape[1] for t in tensors)
        min_w = min(t.shape[2] for t in tensors)
        result = []
        ixt_result = []
        for t, ixt in zip(tensors, ixts):
            h, w = t.shape[1], t.shape[2]
            if h != min_h or w != min_w:
                top = (h - min_h) // 2
                left = (w - min_w) // 2
                t = t[:, top:top + min_h, left:left + min_w]
                if ixt is not None:
                    ixt = ixt.copy()
                    ixt[0, 2] -= left
                    ixt[1, 2] -= top
            result.append(t)
            ixt_result.append(ixt)
        return result, ixt_result

    @staticmethod
    def denormalize(tensor: torch.Tensor) -> np.ndarray:
        """Denormalize a (N, 3, H, W) tensor to (N, H, W, 3) uint8 numpy."""
        mean = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
        std = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)
        imgs = tensor * std + mean
        imgs = imgs.clamp(0, 1).permute(0, 2, 3, 1).numpy()
        return (imgs * 255).astype(np.uint8)


if HAS_REGISTRY:
    TRANSFORMS.register_module()(DA3InputProcessor)
