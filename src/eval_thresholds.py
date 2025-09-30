"""Command-line utility to sweep detection thresholds for count accuracy."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from . import config
from .dataset import XViewVehicleDataset, collate_fn
from .eval import save_calibrated_thresholds, sweep_thresholds
from .model import create_model
from .transforms import build_val_transforms
from .utils import set_seed, setup_logging
from .splits import create_grouped_indices


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Threshold sweep for vehicle counts")
    parser.add_argument("--model_path", type=Path, required=True, help="Path to trained model checkpoint")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=config.NUM_WORKERS)
    parser.add_argument("--output", type=Path, default=config.CALIBRATED_THRESHOLDS_FILE)
    parser.add_argument("--search_grid", type=float, nargs="*", default=config.THRESHOLD_SEARCH_GRID)
    parser.add_argument("--use_ema", action="store_true", help="Use EMA weights if present in checkpoint")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed()
    logger = setup_logging()

    val_transforms = build_val_transforms()
    dataset = XViewVehicleDataset(
        images_dir=config.TRAIN_IMAGES_DIR,
        annotations_file=config.ANNOTATIONS_FILE,
        transforms=val_transforms,
    )

    _, val_indices = create_grouped_indices(dataset)
    val_subset = torch.utils.data.Subset(dataset, val_indices)

    data_loader = DataLoader(
        val_subset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=config.PIN_MEMORY,
    )

    model = create_model()
    checkpoint = torch.load(args.model_path, map_location=config.DEVICE)
    if args.use_ema and "ema_state_dict" in checkpoint:
        logger.info("Loading EMA weights for calibration")
        model.load_state_dict(checkpoint["ema_state_dict"])
    else:
        model.load_state_dict(checkpoint["model_state_dict"])
    model.to(config.DEVICE)

    logger.info("Sweeping thresholds across %s", args.search_grid)
    thresholds, metrics = sweep_thresholds(model, data_loader, args.search_grid)

    logger.info("Best thresholds: %s", thresholds)
    logger.info("Metrics at best thresholds: %s", metrics)

    save_calibrated_thresholds(thresholds, args.output)
    logger.info("Saved calibrated thresholds to %s", args.output)


if __name__ == "__main__":
    main()


