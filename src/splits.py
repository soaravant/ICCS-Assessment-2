"""Dataset splitting utilities for leakage-safe grouped splits."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import torch

from . import config


def _load_group_mapping(annotations_file: Path) -> Dict[int, str]:
    with Path(annotations_file).open("r", encoding="utf-8") as f:
        coco = json.load(f)

    mapping: Dict[int, str] = {}
    for image_info in coco.get("images", []):
        file_name = image_info.get("file_name", "")
        parts = file_name.split("_")
        base_id = parts[1] if len(parts) > 1 else file_name
        mapping[image_info["id"]] = base_id
    return mapping


def create_grouped_indices(
    dataset,
    annotations_file: Path = config.ANNOTATIONS_FILE,
    train_ratio: float = config.TRAIN_VAL_SPLIT,
    seed: int = config.RANDOM_SEED,
) -> Tuple[List[int], List[int]]:
    """Split dataset indices by base image id to prevent patch leakage."""

    image_to_group = _load_group_mapping(annotations_file)

    group_to_indices: Dict[str, List[int]] = defaultdict(list)
    for idx, record in enumerate(dataset.image_records):
        group_key = image_to_group.get(record["id"])
        group_to_indices[group_key].append(idx)

    groups = list(group_to_indices.keys())
    if not groups:
        raise ValueError("No groups found when preparing dataset split; check annotations.")

    generator = torch.Generator().manual_seed(seed)
    permutation = torch.randperm(len(groups), generator=generator).tolist()

    split_idx = int(len(groups) * train_ratio)
    train_groups = {groups[i] for i in permutation[:split_idx]}
    val_groups = {groups[i] for i in permutation[split_idx:]}

    if not val_groups:
        # Ensure at least one validation group for metrics tracking
        val_group = groups[permutation[-1]]
        val_groups.add(val_group)
        train_groups.discard(val_group)

    train_indices = [idx for group in train_groups for idx in group_to_indices[group]]
    val_indices = [idx for group in val_groups for idx in group_to_indices[group]]

    return train_indices, val_indices


