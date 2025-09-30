"""Inference script to count vehicles in future pass images."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import tifffile
import torch

from . import config
from .model import create_model
from .transforms import build_inference_transforms
from .utils import setup_logging


def load_image(image_path: Path, augment=None) -> torch.Tensor:
    image_np = tifffile.imread(image_path)
    if image_np.ndim == 2:
        image_np = np.stack([image_np] * 3, axis=-1)
    image_np = image_np[..., :3]
    if image_np.dtype != np.float32:
        image_np = image_np.astype(np.float32)

    if image_np.max() > 1.0:
        image_np /= 255.0

    if augment is not None:
        transformed = augment(image=image_np)
        image_np = transformed["image"]

    tensor = torch.from_numpy(image_np).permute(2, 0, 1).contiguous().float()
    return tensor


def parse_args():
    parser = argparse.ArgumentParser(description="Run inference on future pass images")
    parser.add_argument("--model_path", type=Path, required=True, help="Path to trained model weights")
    parser.add_argument("--output_json", type=Path, default=config.OUTPUT_DIR / "future_pass_counts.json")
    parser.add_argument("--use_ema", action="store_true", help="Load EMA weights if available")
    parser.add_argument("--tta", action="store_true", help="Apply simple flip test-time augmentation")
    return parser.parse_args()


def main():
    args = parse_args()
    logger = setup_logging()
    config.ensure_directories()

    logger.info("Loading model from %s", args.model_path)
    model = create_model()
    checkpoint = torch.load(args.model_path, map_location=config.DEVICE)
    if args.use_ema and "ema_state_dict" in checkpoint:
        logger.info("Loading EMA weights")
        model.load_state_dict(checkpoint["ema_state_dict"])
    else:
        model.load_state_dict(checkpoint["model_state_dict"])
    model.to(config.DEVICE)
    model.eval()

    results = []
    infer_transform = build_inference_transforms()

    # Load calibrated thresholds if available
    thresholds = dict(config.SCORE_THRESHOLDS)
    if config.CALIBRATED_THRESHOLDS_FILE.exists():
        try:
            with config.CALIBRATED_THRESHOLDS_FILE.open("r", encoding="utf-8") as f:
                loaded = json.load(f)
            loaded = {int(k): float(v) for k, v in loaded.items()}
            thresholds.update(loaded)
            logger.info("Using calibrated thresholds: %s", thresholds)
        except Exception as e:
            logger.warning("Failed to load calibrated thresholds: %s", e)

    with torch.no_grad():
        for image_path in sorted(config.FUTURE_PASS_DIR.iterdir()):
            if image_path.suffix.lower() != ".tif":
                continue

            image = load_image(image_path, augment=infer_transform).to(config.DEVICE)

            images_to_eval = [image]

            if args.tta:
                images_to_eval.extend([
                    torch.flip(image, dims=[2]),
                    torch.flip(image, dims=[1]),
                ])

            aggregated_counts = {1: [], 2: []}

            for aug_idx, img_tensor in enumerate(images_to_eval):
                outputs = model([img_tensor])
                output = outputs[0]
                scores = output["scores"].detach().cpu()
                labels = output["labels"].detach().cpu()

                small_keep = scores >= thresholds[1]
                large_keep = scores >= thresholds[2]

                small_count = (labels[small_keep] == config.CLASS_ID_MAP[1]).sum().item()
                large_count = (labels[large_keep] == config.CLASS_ID_MAP[2]).sum().item()

                aggregated_counts[1].append(small_count)
                aggregated_counts[2].append(large_count)

            # Median reduces outlier effects across TTA variants
            small_count = int(np.median(aggregated_counts[1])) if aggregated_counts[1] else 0
            large_count = int(np.median(aggregated_counts[2])) if aggregated_counts[2] else 0

            results.append(
                {
                    "image_id": image_path.name,
                    "small_vehicle_count": int(small_count),
                    "large_vehicle_count": int(large_count),
                }
            )

    # Try to load training summary to include in final JSON
    overall_metrics = {}
    if config.TRAINING_SUMMARY_FILE.exists():
        try:
            with config.TRAINING_SUMMARY_FILE.open("r", encoding="utf-8") as f:
                overall_metrics = json.load(f)
        except Exception as e:
            logger.warning("Failed to load training summary: %s", e)

    output_data = {"results": results, "overall_metrics": overall_metrics}

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with args.output_json.open("w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2)

    logger.info("Saved inference results to %s", args.output_json)


if __name__ == "__main__":
    main()


