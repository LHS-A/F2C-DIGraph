"""Synchronized registered CFP--FFA--mask preprocessing."""

from __future__ import annotations

import random
from typing import Dict, List

import numpy as np
import torch
from PIL import Image
from torchvision.transforms import InterpolationMode, RandomCrop
from torchvision.transforms import functional as TF


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class SynchronizedRetinaTransform:
    """Apply identical geometry to every registered image and structure mask."""

    def __init__(self, image_size: int = 512, train: bool = True) -> None:
        self.image_size = image_size
        self.train = train

    def __call__(
        self,
        cfp: Image.Image,
        ffa_frames: List[Image.Image],
        masks: Dict[str, np.ndarray],
        valid_overlap: np.ndarray,
    ) -> tuple[torch.Tensor, torch.Tensor, Dict[str, np.ndarray], torch.Tensor]:
        images = [cfp.convert("RGB"), *(frame.convert("L") for frame in ffa_frames)]
        mask_images = {name: Image.fromarray(mask.astype(np.uint8) * 255, mode="L") for name, mask in masks.items()}
        valid = Image.fromarray(valid_overlap.astype(np.uint8) * 255, mode="L")
        target = self.image_size + 32 if self.train else self.image_size
        images = [TF.resize(image, [target, target], antialias=True) for image in images]
        mask_images = {
            name: TF.resize(mask, [target, target], interpolation=InterpolationMode.NEAREST)
            for name, mask in mask_images.items()
        }
        valid = TF.resize(valid, [target, target], interpolation=InterpolationMode.NEAREST)

        if self.train:
            top, left, height, width = RandomCrop.get_params(images[0], [self.image_size, self.image_size])
            images = [TF.crop(image, top, left, height, width) for image in images]
            mask_images = {name: TF.crop(mask, top, left, height, width) for name, mask in mask_images.items()}
            valid = TF.crop(valid, top, left, height, width)
            if random.random() < 0.5:
                images = [TF.hflip(image) for image in images]
                mask_images = {name: TF.hflip(mask) for name, mask in mask_images.items()}
                valid = TF.hflip(valid)
            if random.random() < 0.5:
                images = [TF.vflip(image) for image in images]
                mask_images = {name: TF.vflip(mask) for name, mask in mask_images.items()}
                valid = TF.vflip(valid)
            angle = random.uniform(-10.0, 10.0)
            images = [TF.rotate(image, angle, interpolation=InterpolationMode.BILINEAR, fill=0) for image in images]
            mask_images = {
                name: TF.rotate(mask, angle, interpolation=InterpolationMode.NEAREST, fill=0)
                for name, mask in mask_images.items()
            }
            valid = TF.rotate(valid, angle, interpolation=InterpolationMode.NEAREST, fill=0)

        cfp_tensor = TF.normalize(TF.to_tensor(images[0]), IMAGENET_MEAN, IMAGENET_STD)
        ffa_tensor = torch.cat([TF.to_tensor(frame) for frame in images[1:]], dim=0)
        ffa_tensor = TF.normalize(ffa_tensor, IMAGENET_MEAN, IMAGENET_STD)
        transformed_masks = {
            name: np.asarray(mask, dtype=np.uint8) > 0 for name, mask in mask_images.items()
        }
        valid_tensor = (TF.to_tensor(valid) > 0.5).float()
        return cfp_tensor, ffa_tensor, transformed_masks, valid_tensor


def build_transforms(image_size: int = 512, train: bool = True) -> SynchronizedRetinaTransform:
    return SynchronizedRetinaTransform(image_size, train)
