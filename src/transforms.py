"""Data augmentations and normalization utilities."""

from __future__ import annotations

from typing import Dict

import albumentations as A
import cv2
import numpy as np
import torch

from . import config


def build_train_transforms(fast: bool = False) -> A.Compose:
    if fast:
        aug = [
            A.HorizontalFlip(p=0.5),
            A.Resize(config.TRAIN_IMAGE_SIZE, config.TRAIN_IMAGE_SIZE, interpolation=cv2.INTER_LINEAR),
            A.Normalize(mean=config.IMAGENET_MEAN, std=config.IMAGENET_STD, max_pixel_value=1.0),
        ]
    else:
        aug = [
            A.RandomRotate90(p=0.5),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.2),
            A.Affine(
                scale=(0.85, 1.15),
                translate_percent=0.12,
                rotate=(-10, 10),
                shear=(-5, 5),
                interpolation=cv2.INTER_LINEAR,
                p=0.5,
                mode=cv2.BORDER_REFLECT_101,
            ),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05, p=0.4),
            A.GaussNoise(var_limit=(10.0, 60.0), mean=0, p=0.2),
            A.Resize(config.TRAIN_IMAGE_SIZE, config.TRAIN_IMAGE_SIZE, interpolation=cv2.INTER_LINEAR),
            A.Normalize(mean=config.IMAGENET_MEAN, std=config.IMAGENET_STD, max_pixel_value=1.0),
        ]
    return A.Compose(
        aug,
        bbox_params=A.BboxParams(
            format="pascal_voc",
            label_fields=["labels"],
            min_visibility=0.25,
        ),
    )


def build_val_transforms() -> A.Compose:
    return A.Compose(
        [
            A.Resize(config.VAL_IMAGE_SIZE, config.VAL_IMAGE_SIZE, interpolation=cv2.INTER_LINEAR),
            A.Normalize(mean=config.IMAGENET_MEAN, std=config.IMAGENET_STD, max_pixel_value=1.0),
        ],
        bbox_params=A.BboxParams(format="pascal_voc", label_fields=["labels"]),
    )


def build_inference_transforms() -> A.Compose:
    return A.Compose(
        [
            A.Resize(config.INFER_IMAGE_SIZE, config.INFER_IMAGE_SIZE, interpolation=cv2.INTER_LINEAR),
            A.Normalize(mean=config.IMAGENET_MEAN, std=config.IMAGENET_STD, max_pixel_value=1.0),
        ]
    )


def apply_transforms(image: torch.Tensor, target: Dict[str, torch.Tensor], transform: A.Compose):
    image_np = image.permute(1, 2, 0).cpu().numpy()
    boxes = target["boxes"].cpu().numpy() if target["boxes"].numel() > 0 else np.zeros((0, 4), dtype=np.float32)
    labels = target["labels"].cpu().numpy() if target["labels"].numel() > 0 else np.zeros((0,), dtype=np.int64)

    if boxes.size == 0:
        augmented = transform(image=image_np, bboxes=[], labels=[])
        aug_boxes = np.zeros((0, 4), dtype=np.float32)
        aug_labels = np.zeros((0,), dtype=np.int64)
    else:
        augmented = transform(image=image_np, bboxes=boxes.tolist(), labels=labels.tolist())
        aug_boxes = np.array(augmented["bboxes"], dtype=np.float32)
        aug_labels = np.array(augmented["labels"], dtype=np.int64)

    if aug_boxes.ndim == 1:
        aug_boxes = aug_boxes.reshape(-1, 4)

    transformed_image = torch.from_numpy(augmented["image"]).permute(2, 0, 1).contiguous().float()

    target["boxes"] = torch.from_numpy(aug_boxes).float()
    target["labels"] = torch.from_numpy(aug_labels).long()

    return transformed_image, target


