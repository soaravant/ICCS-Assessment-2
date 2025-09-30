"""Dataset utilities for xView vehicle detection."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import numpy as np
import tifffile
import torch
from torch.utils.data import Dataset

from . import config
from . import transforms as custom_transforms


@dataclass(frozen=True)
class Annotation:
    bbox: Tuple[float, float, float, float]
    category_id: int
    area: float
    iscrowd: int


class XViewVehicleDataset(Dataset):
    """PyTorch dataset for xView vehicle detection (small vs large vehicles)."""

    def __init__(
        self,
        images_dir: Path,
        annotations_file: Path,
        transforms: Optional[Callable] = None,
        include_categories: Optional[Iterable[int]] = None,
    ) -> None:
        self.images_dir = Path(images_dir)
        self.annotations_file = Path(annotations_file)
        self.transforms = transforms
        self.include_categories = set(include_categories or config.CLASS_ID_MAP.keys())

        if not self.images_dir.exists():
            raise FileNotFoundError(f"Images directory not found: {self.images_dir}")
        if not self.annotations_file.exists():
            raise FileNotFoundError(f"Annotations file not found: {self.annotations_file}")

        with self.annotations_file.open("r", encoding="utf-8") as f:
            coco = json.load(f)

        self.categories = {c["id"]: c["name"] for c in coco.get("categories", [])}

        self.image_records = coco.get("images", [])
        self.id_to_record = {img["id"]: img for img in self.image_records}

        ann_by_image: Dict[int, List[Annotation]] = {}
        for ann in coco.get("annotations", []):
            category_id = ann.get("category_id")
            if category_id not in self.include_categories:
                continue
            bbox = ann.get("bbox", [0, 0, 0, 0])
            x, y, w, h = bbox
            # Skip degenerate boxes
            if w <= 1 or h <= 1:
                continue
            ann_struct = Annotation(
                bbox=(float(x), float(y), float(x + w), float(y + h)),
                category_id=config.CLASS_ID_MAP[category_id],
                area=float(ann.get("area", w * h)),
                iscrowd=int(ann.get("iscrowd", 0)),
            )
            ann_by_image.setdefault(ann.get("image_id"), []).append(ann_struct)

        self.ann_by_image = ann_by_image

    def __len__(self) -> int:
        return len(self.image_records)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        image_record = self.image_records[idx]
        image_id = image_record["id"]
        file_name = image_record["file_name"]
        img_path = self.images_dir / file_name

        if not img_path.exists():
            raise FileNotFoundError(f"Image not found: {img_path}")

        image_np = tifffile.imread(img_path)
        if image_np.ndim == 2:
            image_np = np.stack([image_np] * 3, axis=-1)
        image_np = image_np[..., :3]
        # convert to float32 in [0,1]
        if image_np.dtype != np.float32:
            image_np = image_np.astype(np.float32)
        if image_np.max() > 1.0:
            image_np /= 255.0
        image_tensor = torch.from_numpy(image_np).permute(2, 0, 1)

        annotations = self.ann_by_image.get(image_id, [])
        boxes = torch.zeros((len(annotations), 4), dtype=torch.float32)
        labels = torch.zeros((len(annotations),), dtype=torch.int64)
        areas = torch.zeros((len(annotations),), dtype=torch.float32)
        iscrowd = torch.zeros((len(annotations),), dtype=torch.int64)

        for i, ann in enumerate(annotations):
            boxes[i] = torch.tensor(ann.bbox, dtype=torch.float32)
            labels[i] = ann.category_id
            areas[i] = ann.area
            iscrowd[i] = ann.iscrowd

        target: Dict[str, torch.Tensor] = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([image_id]),
            "area": areas,
            "iscrowd": iscrowd,
        }

        if self.transforms:
            image_tensor, target = custom_transforms.apply_transforms(image_tensor, target, self.transforms)
        else:
            # If no transforms are provided, normalize to ImageNet stats for backbone compatibility
            mean = torch.tensor(config.IMAGENET_MEAN, dtype=image_tensor.dtype).view(3, 1, 1)
            std = torch.tensor(config.IMAGENET_STD, dtype=image_tensor.dtype).view(3, 1, 1)
            image_tensor = (image_tensor - mean) / std

        return image_tensor, target

    def __repr__(self) -> str:
        return (
            f"XViewVehicleDataset(num_images={len(self)}, include_categories={sorted(self.include_categories)})"
        )


def collate_fn(batch: List[Tuple[torch.Tensor, Dict[str, torch.Tensor]]]):
    """Custom collate function for variable-length targets."""

    images, targets = zip(*batch)
    return list(images), list(targets)


