"""Device-specific training preset management."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence
import sys

import torch


@dataclass(frozen=True)
class TrainingPreset:
    """Container for device-tuned defaults that remain optional."""

    name: str
    description: str
    device: Optional[str] = None
    batch_size: Optional[int] = None
    num_workers: Optional[int] = None
    prefetch_factor: Optional[int] = None
    pin_memory: Optional[bool] = None
    accumulate: Optional[int] = None
    img_size: Optional[int] = None
    fast_aug: Optional[bool] = None
    fast_mode: Optional[bool] = None
    trainable_layers: Optional[int] = None
    lr: Optional[float] = None
    max_steps: Optional[int] = None


def _registry() -> Dict[str, TrainingPreset]:
    return {
        "m3": TrainingPreset(
            name="m3",
            description="Apple M3 Max tuned for balanced throughput without overheating",
            device="mps",
            batch_size=4,
            num_workers=4,
            prefetch_factor=1,
            pin_memory=False,
            accumulate=2,
            img_size=640,
            fast_aug=True,
            fast_mode=True,
            trainable_layers=2,
            max_steps=80,
        ),
        "cuda_5060": TrainingPreset(
            name="cuda_5060",
            description="GeForce RTX 5060 preset for full accuracy runs",
            device="cuda",
            batch_size=6,
            num_workers=8,
            prefetch_factor=4,
            pin_memory=True,
            accumulate=1,
            img_size=896,
            fast_aug=False,
            fast_mode=False,
            trainable_layers=4,
        ),
        "cpu": TrainingPreset(
            name="cpu",
            description="CPU fallback preset",
            device="cpu",
            batch_size=2,
            num_workers=2,
            prefetch_factor=2,
            pin_memory=False,
            accumulate=2,
            img_size=512,
            fast_aug=True,
            fast_mode=True,
            trainable_layers=1,
            max_steps=60,
        ),
    }


def available_presets() -> Sequence[str]:
    """Expose CLI choices."""

    return ("auto",) + tuple(_registry().keys())


def resolve(name: str) -> TrainingPreset:
    """Resolve preset name, handling auto-detection."""

    name = name.lower()
    registry = _registry()
    if name == "auto":
        if torch.cuda.is_available():
            return registry["cuda_5060"]
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return registry["m3"]
        return registry["cpu"]

    if name not in registry:
        known = ", ".join(sorted(registry))
        raise ValueError(f"Unknown preset '{name}'. Available presets: {known} or 'auto'.")

    return registry[name]


def _flag_used(flag: str, cli_args: Sequence[str]) -> bool:
    prefix = flag + "="
    for token in cli_args:
        if token == flag or token.startswith(prefix):
            return True
    return False


def apply(args, preset: TrainingPreset, cli_args: Optional[Sequence[str]] = None) -> None:
    """Fill unset CLI args with preset defaults while respecting user overrides."""

    cli_args = tuple(sys.argv[1:] if cli_args is None else cli_args)

    def assign(attr: str, value: Optional[object], *flags: str) -> None:
        if value is None:
            return
        if any(_flag_used(flag, cli_args) for flag in flags):
            return
        setattr(args, attr, value)

    assign("device", preset.device, "--device")
    assign("batch_size", preset.batch_size, "--batch_size")
    assign("num_workers", preset.num_workers, "--num_workers")
    assign("prefetch_factor", preset.prefetch_factor, "--prefetch_factor")
    assign("pin_memory", preset.pin_memory, "--pin_memory", "--pin-memory")
    assign("accumulate", preset.accumulate, "--accumulate")
    assign("img_size", preset.img_size, "--img_size")
    assign("fast_aug", preset.fast_aug, "--fast_aug", "--fast-aug")
    assign("fast_mode", preset.fast_mode, "--fast_mode", "--fast-mode")
    assign("trainable_layers", preset.trainable_layers, "--trainable_layers")
    assign("lr", preset.lr, "--lr")
    assign("max_steps", preset.max_steps, "--max_steps")

    # Track the resolved preset name for logging/debugging downstream
    args.preset = preset.name
