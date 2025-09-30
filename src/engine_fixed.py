"""Training and evaluation loops for object detection models."""

from __future__ import annotations

from dataclasses import dataclass, field
import time
from typing import Dict, Iterable, List, Optional

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.ops import box_iou
from tqdm import tqdm

from . import config


@dataclass
class EpochMetrics:
    loss: float = 0.0
    loss_classifier: float = 0.0
    loss_box_reg: float = 0.0
    loss_objectness: float = 0.0
    loss_rpn_box_reg: float = 0.0
    num_batches: int = 0

    def update(self, loss_dict: Dict[str, torch.Tensor]):
        self.loss += loss_dict["loss"].item()
        self.loss_classifier += loss_dict["loss_classifier"].item()
        self.loss_box_reg += loss_dict["loss_box_reg"].item()
        self.loss_objectness += loss_dict["loss_objectness"].item()
        self.loss_rpn_box_reg += loss_dict["loss_rpn_box_reg"].item()
        self.num_batches += 1

    def average(self) -> Dict[str, float]:
        if self.num_batches == 0:
            return {}
        return {
            "loss": self.loss / self.num_batches,
            "loss_classifier": self.loss_classifier / self.num_batches,
            "loss_box_reg": self.loss_box_reg / self.num_batches,
            "loss_objectness": self.loss_objectness / self.num_batches,
            "loss_rpn_box_reg": self.loss_rpn_box_reg / self.num_batches,
        }


@torch.no_grad()
def evaluate_model(model: nn.Module, data_loader: DataLoader) -> Dict[str, float]:
    model.eval()

    total_iou = 0.0
    total_precision = 0.0
    total_recall = 0.0
    total_small_mae = 0.0
    total_large_mae = 0.0
    total_small_rmse = 0.0
    total_large_rmse = 0.0
    num_samples = 0

    progress_bar = tqdm(data_loader, desc="Validation", leave=False)
    for images, targets in progress_bar:
        images = [img.to(config.DEVICE) for img in images]
        outputs = model(images)

        for prediction, target in zip(outputs, targets):
            gt_boxes = target["boxes"].to(config.DEVICE)
            gt_labels = target["labels"].to(config.DEVICE)
            det_boxes_all = prediction["boxes"].to(config.DEVICE)
            det_scores_all = prediction["scores"].to(config.DEVICE)
            det_labels_all = prediction["labels"].to(config.DEVICE)

            th_small = config.SCORE_THRESHOLDS[1]
            th_large = config.SCORE_THRESHOLDS[2]
            keep_mask = ((det_labels_all == config.CLASS_ID_MAP[1]) & (det_scores_all >= th_small)) | (
                (det_labels_all == config.CLASS_ID_MAP[2]) & (det_scores_all >= th_large)
            )
            det_boxes = det_boxes_all[keep_mask]
            det_labels = det_labels_all[keep_mask]
            det_scores = det_scores_all[keep_mask]

            gt_small = (gt_labels == config.CLASS_ID_MAP[1]).sum().item()
            gt_large = (gt_labels == config.CLASS_ID_MAP[2]).sum().item()
            pred_small = (det_labels == config.CLASS_ID_MAP[1]).sum().item()
            pred_large = (det_labels == config.CLASS_ID_MAP[2]).sum().item()

            total_small_mae += abs(pred_small - gt_small)
            total_large_mae += abs(pred_large - gt_large)
            total_small_rmse += (pred_small - gt_small) ** 2
            total_large_rmse += (pred_large - gt_large) ** 2

            if det_boxes.numel() == 0 or gt_boxes.numel() == 0:
                num_samples += 1
                continue

            ious = box_iou(det_boxes, gt_boxes)
            max_iou, _ = ious.max(dim=1)

            matched = max_iou > 0.5
            tp = matched.sum().item()
            fp = det_boxes.size(0) - tp
            fn = gt_boxes.size(0) - tp

            total_iou += max_iou[matched].sum().item()
            total_precision += tp / (tp + fp + 1e-6)
            total_recall += tp / (tp + fn + 1e-6)
            num_samples += 1

    if num_samples == 0:
        return {
            "iou": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "small_mae": 0.0,
            "large_mae": 0.0,
            "small_rmse": 0.0,
            "large_rmse": 0.0,
        }

    return {
        "iou": total_iou / max(1, num_samples),
        "precision": total_precision / max(1, num_samples),
        "recall": total_recall / max(1, num_samples),
        "small_mae": total_small_mae / max(1, num_samples),
        "large_mae": total_large_mae / max(1, num_samples),
        "small_rmse": (total_small_rmse / max(1, num_samples)) ** 0.5,
        "large_rmse": (total_large_rmse / max(1, num_samples)) ** 0.5,
    }


def train_one_epoch(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    data_loader: DataLoader,
    lr_scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
    epoch: int = 0,
    grad_accum_steps: int = 1,
    total_epochs: Optional[int] = None,
    max_steps: Optional[int] = None,
) -> Dict[str, float]:
    model.train()
    metrics = EpochMetrics()
    optimizer.zero_grad()

    # Fix: Use correct total for progress bar
    total_batches = min(len(data_loader), max_steps) if max_steps is not None else len(data_loader)
    
    progress_bar = tqdm(
        enumerate(data_loader),
        total=total_batches,
        desc=f"Epoch {epoch + 1}/{total_epochs if total_epochs is not None else '?'} [00:00:00]",
        leave=True,
        dynamic_ncols=True,
        mininterval=0.5,
    )

    start_time = time.perf_counter()
    num_images_seen = 0

    def _format_elapsed(seconds: float) -> str:
        s = int(seconds)
        h = s // 3600
        m = (s % 3600) // 60
        s = s % 60
        return f"{h:02d}:{m:02d}:{s:02d}"

    for step, (images, targets) in progress_bar:
        images = [image.to(config.DEVICE) for image in images]
        targets = [{k: v.to(config.DEVICE) for k, v in t.items()} for t in targets]
        num_images_seen += len(images)

        if scaler is not None:
            with torch.cuda.amp.autocast():
                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values()) / grad_accum_steps
            scaler.scale(losses).backward()
        else:
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values()) / grad_accum_steps
            losses.backward()

        if (step + 1) % grad_accum_steps == 0:
            if scaler is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRAD_CLIP_NORM)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRAD_CLIP_NORM)
                optimizer.step()

            optimizer.zero_grad()

            if lr_scheduler is not None:
                lr_scheduler.step()

        with torch.no_grad():
            loss_copy = {k: v.detach() for k, v in loss_dict.items()}
            loss_copy["loss"] = sum(loss_dict.values()).detach()
            metrics.update(loss_copy)
            progress_bar.set_postfix({"loss": float(loss_copy["loss"])})
            elapsed = time.perf_counter() - start_time
            progress_bar.set_description(
                f"Epoch {epoch + 1}/{total_epochs if total_epochs is not None else '?'} [{_format_elapsed(elapsed)}]"
            )
            progress_bar.refresh()

        if max_steps is not None and (step + 1) >= max_steps:
            progress_bar.close()
            break

    avg = metrics.average()
    elapsed = time.perf_counter() - start_time
    if elapsed <= 0:
        elapsed = 1e-6
    avg.update(
        {
            "elapsed_s": float(elapsed),
            "imgs_per_s": float(num_images_seen / elapsed),
            "lr": float(optimizer.param_groups[0]["lr"]) if optimizer.param_groups else 0.0,
        }
    )
    return avg


