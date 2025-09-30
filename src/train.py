"""Training script for xView vehicle detection."""

from __future__ import annotations

import argparse
from argparse import BooleanOptionalAction
from pathlib import Path

import json
import math

import torch
import torch.utils.data
from torch.utils.data import DataLoader

from . import config
from . import presets
from .dataset import XViewVehicleDataset, collate_fn
from .engine import evaluate_model, train_one_epoch
from .model import create_model
from .splits import create_grouped_indices
from .transforms import build_train_transforms, build_val_transforms
from .utils import set_seed, setup_logging


def get_args():
    parser = argparse.ArgumentParser(description="Train vehicle detector on xView dataset")
    parser.add_argument(
        "--preset",
        type=str,
        default="auto",
        choices=presets.available_presets(),
        help="Device preset to use (auto, m3, cuda_5060, cpu)",
    )
    parser.add_argument("--epochs", type=int, default=config.NUM_EPOCHS)
    parser.add_argument("--batch_size", type=int, default=config.BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=config.LEARNING_RATE)
    parser.add_argument("--weight_decay", type=float, default=config.WEIGHT_DECAY)
    parser.add_argument("--num_workers", type=int, default=config.NUM_WORKERS)
    parser.add_argument("--prefetch_factor", type=int, default=config.PREFETCH_FACTOR)
    parser.add_argument(
        "--pin_memory",
        action=BooleanOptionalAction,
        default=None,
        help="Enable or disable dataloader pin-memory",
    )
    parser.add_argument("--output_dir", type=Path, default=config.OUTPUT_DIR)
    parser.add_argument("--model_path", type=Path, default=config.MODELS_DIR / "xview_vehicle_detector.pt")
    parser.add_argument("--no-ema", action="store_true")
    parser.add_argument("--accumulate", type=int, default=config.GRAD_ACCUM_STEPS)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--img_size", type=int, default=None, help="Override train/val image size")
    parser.add_argument("--trainable_layers", type=int, default=2, help="ResNet trainable layers (0-5)")
    parser.add_argument(
        "--fast_aug",
        action=BooleanOptionalAction,
        default=None,
        help="Use faster, lighter augmentations",
    )
    parser.add_argument("--val_every", type=int, default=1, help="Validate every N epochs")
    parser.add_argument("--device", type=str, default=None, help="Override device: cpu|mps|cuda")
    parser.add_argument("--max_steps", type=int, default=None, help="Max train steps per epoch (speed-up)")
    parser.add_argument("--train_ratio", type=float, default=None, help="Fraction of groups for training (0-1)")
    parser.add_argument(
        "--fast_mode",
        action=BooleanOptionalAction,
        default=None,
        help="Reduce proposals/detections for speed",
    )
    return parser.parse_args()


def main():
    args = get_args()
    preset_cfg = presets.resolve(args.preset)
    presets.apply(args, preset_cfg)

    if args.pin_memory is None:
        args.pin_memory = config.PIN_MEMORY
    if args.fast_aug is None:
        args.fast_aug = False
    if args.fast_mode is None:
        args.fast_mode = False

    config.PIN_MEMORY = args.pin_memory
    config.NUM_WORKERS = args.num_workers
    config.PREFETCH_FACTOR = args.prefetch_factor
    config.BATCH_SIZE = args.batch_size

    set_seed()
    config.ensure_directories()
    logger = setup_logging()

    logger.info("Using preset '%s': %s", args.preset, preset_cfg.description)

    # Optional device override
    if args.device is not None:
        dev = args.device.lower()
        if dev == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA requested but not available; staying on %s", config.DEVICE)
        elif dev == "mps" and not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
            logger.warning("MPS requested but not available; staying on %s", config.DEVICE)
        else:
            config.DEVICE = torch.device(dev)
            logger.info("Using device override: %s", config.DEVICE)

    logger.info("Loading dataset...")
    # Optionally override image sizes for faster runs
    if args.img_size is not None:
        config.TRAIN_IMAGE_SIZE = args.img_size
        config.VAL_IMAGE_SIZE = args.img_size
    config.GRAD_ACCUM_STEPS = args.accumulate
    train_transforms = build_train_transforms(fast=args.fast_aug)
    val_transforms = build_val_transforms()

    dataset = XViewVehicleDataset(
        images_dir=config.TRAIN_IMAGES_DIR,
        annotations_file=config.ANNOTATIONS_FILE,
        transforms=train_transforms,
    )

    train_ratio = float(args.train_ratio) if args.train_ratio is not None else config.TRAIN_VAL_SPLIT
    train_indices, val_indices = create_grouped_indices(dataset, train_ratio=train_ratio)

    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    val_base_dataset = XViewVehicleDataset(
        images_dir=config.TRAIN_IMAGES_DIR,
        annotations_file=config.ANNOTATIONS_FILE,
        transforms=val_transforms,
    )
    val_dataset = torch.utils.data.Subset(val_base_dataset, val_indices)

    logger.info("Creating data loaders...")
    common_loader_kwargs = dict(
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=args.pin_memory,
    )
    if args.num_workers > 0:
        common_loader_kwargs["prefetch_factor"] = args.prefetch_factor
        common_loader_kwargs["persistent_workers"] = True
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        **common_loader_kwargs,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        **common_loader_kwargs,
    )
    loader_prefetch = common_loader_kwargs.get("prefetch_factor", "n/a")
    loader_persistent = common_loader_kwargs.get("persistent_workers", False)
    logger.info(
        "Loader config: batch=%s accumulate=%s workers=%s pin_memory=%s prefetch=%s persistent=%s",
        args.batch_size,
        args.accumulate,
        args.num_workers,
        args.pin_memory,
        loader_prefetch,
        loader_persistent,
    )

    logger.info("Creating model...")
    if args.fast_mode:
        # Strongly limit proposals/detections for speed
        config.RPN_PRE_NMS_TOP_N_TRAIN = 300
        config.RPN_PRE_NMS_TOP_N_TEST = 300
        config.RPN_POST_NMS_TOP_N_TRAIN = 100
        config.RPN_POST_NMS_TOP_N_TEST = 100
        config.MAX_DETECTIONS_PER_IMAGE = 100
    # Switch to MobileNetV3 320 FPN for much faster iterations on MPS
    model = create_model(trainable_layers=max(0, min(5, args.trainable_layers)), arch="mbv3_320")
    model.to(config.DEVICE)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)

    total_steps = len(train_loader) * args.epochs
    warmup_steps = len(train_loader) * config.WARMUP_EPOCHS

    def lr_lambda(current_step: int):
        if current_step < warmup_steps:
            return float(current_step) / max(1, warmup_steps)
        progress = (current_step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(config.COSINE_FINAL_LR / args.lr, 0.5 * (1 + math.cos(math.pi * progress)))

    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None

    history = {"train": [], "val": []}
    ema_state = None

    if args.resume:
        logger.info("Resuming training from %s", args.resume)
        checkpoint = torch.load(args.resume, map_location=config.DEVICE)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        history = checkpoint.get("history", history)
        start_epoch = checkpoint.get("epoch", 0)
        if "ema_state_dict" in checkpoint and not args.no_ema:
            ema_state = checkpoint["ema_state_dict"]
        lr_scheduler.last_epoch = start_epoch * len(train_loader)
    else:
        start_epoch = 0

    for epoch in range(start_epoch, args.epochs):
        logger.info("Epoch %d/%d", epoch + 1, args.epochs)
        train_metrics = train_one_epoch(
            model,
            optimizer,
            train_loader,
            lr_scheduler=lr_scheduler,
            scaler=scaler,
            epoch=epoch,
            grad_accum_steps=args.accumulate,
            total_epochs=args.epochs,
            max_steps=args.max_steps,
        )

        if not args.no_ema:
            if ema_state is None:
                ema_state = {name: param.detach().clone() for name, param in model.state_dict().items()}
            else:
                with torch.no_grad():
                    for name, param in model.state_dict().items():
                        ema_state[name].mul_(config.EMA_DECAY).add_(param.detach(), alpha=1 - config.EMA_DECAY)

        val_metrics = evaluate_model(model, val_loader) if ((epoch + 1) % args.val_every == 0) else {"skipped": True}

        logger.info("Train metrics: %s", train_metrics)
        logger.info("Val metrics: %s", val_metrics)

        history["train"].append(train_metrics)
        history["val"].append(val_metrics)

        # Write history incrementally so the monitor updates live
        try:
            tmp_path = config.TRAINING_HISTORY_FILE.with_suffix(".json.tmp")
            with tmp_path.open("w", encoding="utf-8") as f:
                json.dump(history, f, indent=2)
            tmp_path.replace(config.TRAINING_HISTORY_FILE)
        except Exception as e:
            logger.warning("Failed to write incremental training history: %s", e)

        checkpoint = {
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "history": history,
        }

        if ema_state is not None:
            checkpoint["ema_state_dict"] = ema_state

        torch.save(checkpoint, args.model_path)
        logger.info("Saved model checkpoint to %s", args.model_path)

    # Save full history at the end as well
    try:
        with config.TRAINING_HISTORY_FILE.open("w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)
    except Exception as e:
        logger.warning("Failed to write final training history: %s", e)

    # Write a compact training summary for inclusion in inference JSON
    try:
        import numpy as np
        train_keys = list(history["train"][0].keys()) if history["train"] else []
        train_summary = {
            k: float(np.mean([e[k] for e in history["train"][-5:]])) for k in train_keys
        } if train_keys else {}
        val_summary = history["val"][-1] if history["val"] else {}
        summary = {"train": train_summary, "val": val_summary}
        with config.TRAINING_SUMMARY_FILE.open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        logger.info("Saved training summary to %s", config.TRAINING_SUMMARY_FILE)
    except Exception as e:
        logger.warning("Failed to write training summary: %s", e)


if __name__ == "__main__":
    main()
