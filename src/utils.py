"""Utility functions for training, evaluation, and logging."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import json
import logging
import random

import numpy as np
import torch

from . import config


def set_seed(seed: int = config.RANDOM_SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup_logging() -> logging.Logger:
    logger = logging.getLogger("xview")
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger


def save_metrics(metrics: Dict[str, Dict[str, float]], output_path: Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)


def save_predictions(predictions: List[Dict], output_path: Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(predictions, f, indent=2)


def move_to_device(batch, device):
    if isinstance(batch, torch.Tensor):
        return batch.to(device)
    if isinstance(batch, list):
        return [move_to_device(x, device) for x in batch]
    if isinstance(batch, dict):
        return {k: move_to_device(v, device) for k, v in batch.items()}
    return batch


def reduce_dict(input_dict: Dict[str, torch.Tensor], average: bool = True) -> Dict[str, torch.Tensor]:
    world_size = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
    if world_size < 2:
        return input_dict

    with torch.no_grad():
        keys = sorted(input_dict.keys())
        values = torch.stack([input_dict[k] for k in keys], dim=0)
        torch.distributed.all_reduce(values)
        if average:
            values /= world_size
        return {k: v for k, v in zip(keys, values)}


