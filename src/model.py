"""Model factory for xView vehicle detection."""

from __future__ import annotations

import torch
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_320_fpn
from torchvision.models.detection import FasterRCNN_MobileNet_V3_Large_320_FPN_Weights
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.rpn import AnchorGenerator

from . import config


def create_model(
    num_classes: int = config.NUM_CLASSES,
    pretrained: bool = True,
    trainable_layers: int = 3,
    arch: str = "resnet50_fpn",
) -> FasterRCNN:
    """Create a Faster R-CNN model fine-tuned for vehicle detection.

    Args:
        num_classes: number of classes including background.
        pretrained: load ImageNet-pretrained backbone weights.
        trainable_layers: how many trainable backbone layers. Lower is faster on MPS.
        arch: 'resnet50_fpn' (default) or 'mbv3_320' for a faster MobileNetV3 320 FPN.
    """

    arch = (arch or "resnet50_fpn").lower()

    if arch == "mbv3_320":
        weights = FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.DEFAULT if pretrained else None
        model = fasterrcnn_mobilenet_v3_large_320_fpn(
            weights=weights,
            trainable_backbone_layers=max(0, min(5, trainable_layers)),
            box_detections_per_img=config.MAX_DETECTIONS_PER_IMAGE,
            rpn_pre_nms_top_n_train=config.RPN_PRE_NMS_TOP_N_TRAIN,
            rpn_pre_nms_top_n_test=config.RPN_PRE_NMS_TOP_N_TEST,
            rpn_post_nms_top_n_train=config.RPN_POST_NMS_TOP_N_TRAIN,
            rpn_post_nms_top_n_test=config.RPN_POST_NMS_TOP_N_TEST,
        )
    else:
        backbone = resnet_fpn_backbone(
            "resnet50",
            weights="DEFAULT" if pretrained else None,
            trainable_layers=trainable_layers,
        )

        anchor_sizes = ((4,), (8,), (16,), (32,), (64,))
        aspect_ratios = ((0.5, 1.0, 2.0, 3.0),) * len(anchor_sizes)
        rpn_anchor_generator = AnchorGenerator(anchor_sizes, aspect_ratios)

        model = FasterRCNN(
            backbone,
            num_classes=num_classes,
            rpn_anchor_generator=rpn_anchor_generator,
            box_detections_per_img=config.MAX_DETECTIONS_PER_IMAGE,
            rpn_pre_nms_top_n_train=config.RPN_PRE_NMS_TOP_N_TRAIN,
            rpn_pre_nms_top_n_test=config.RPN_PRE_NMS_TOP_N_TEST,
            rpn_post_nms_top_n_train=config.RPN_POST_NMS_TOP_N_TRAIN,
            rpn_post_nms_top_n_test=config.RPN_POST_NMS_TOP_N_TEST,
        )

    # Replace predictor to match our num_classes
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model


