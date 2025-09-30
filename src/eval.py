"""Evaluation helpers for reporting metrics and threshold tuning."""

from __future__ import annotations

import json
from itertools import product
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from . import config


def evaluate_counts_at_thresholds(
    model,
    data_loader: DataLoader,
    thresholds: Dict[int, float],
    device: torch.device = config.DEVICE,
) -> Dict[str, float]:
    model.eval()
    mae_small = []
    mae_large = []
    rmse_small = []
    rmse_large = []

    with torch.no_grad():
        for images, targets in data_loader:
            images = [img.to(device) for img in images]
            outputs = model(images)

            for prediction, target in zip(outputs, targets):
                scores = prediction["scores"].cpu()
                labels = prediction["labels"].cpu()

                gt_labels = target["labels"].cpu()
                gt_small = (gt_labels == config.CLASS_ID_MAP[1]).sum().item()
                gt_large = (gt_labels == config.CLASS_ID_MAP[2]).sum().item()

                pred_small = ((labels == config.CLASS_ID_MAP[1]) & (scores >= thresholds[1])).sum().item()
                pred_large = ((labels == config.CLASS_ID_MAP[2]) & (scores >= thresholds[2])).sum().item()

                mae_small.append(abs(pred_small - gt_small))
                mae_large.append(abs(pred_large - gt_large))
                rmse_small.append((pred_small - gt_small) ** 2)
                rmse_large.append((pred_large - gt_large) ** 2)

    return {
        "mae_small": float(np.mean(mae_small)) if mae_small else 0.0,
        "mae_large": float(np.mean(mae_large)) if mae_large else 0.0,
        "rmse_small": float(np.sqrt(np.mean(rmse_small))) if rmse_small else 0.0,
        "rmse_large": float(np.sqrt(np.mean(rmse_large))) if rmse_large else 0.0,
    }


def sweep_thresholds(
    model,
    data_loader: DataLoader,
    search_grid: Sequence[float] = config.THRESHOLD_SEARCH_GRID,
    device: torch.device = config.DEVICE,
) -> Tuple[Dict[int, float], Dict[str, float]]:
    """Exhaustive grid search to minimize combined MAE for small and large vehicles."""

    best_thresholds = dict(config.SCORE_THRESHOLDS)
    best_metrics = evaluate_counts_at_thresholds(model, data_loader, best_thresholds, device=device)
    best_score = best_metrics["mae_small"] + best_metrics["mae_large"]

    for small_thresh, large_thresh in product(search_grid, repeat=2):
        thresholds = {1: small_thresh, 2: large_thresh}
        metrics = evaluate_counts_at_thresholds(model, data_loader, thresholds, device=device)
        score = metrics["mae_small"] + metrics["mae_large"]
        if score < best_score:
            best_score = score
            best_thresholds = thresholds
            best_metrics = metrics

    return best_thresholds, best_metrics


def write_overall_metrics(metrics: Dict[str, Dict[str, float]], output_path: Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)


def save_calibrated_thresholds(thresholds: Dict[int, float], output_path: Path = config.CALIBRATED_THRESHOLDS_FILE) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump({str(k): v for k, v in thresholds.items()}, f, indent=2)


