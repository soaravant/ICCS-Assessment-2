"""Project configuration and hyperparameters for xView vehicle detection."""

from pathlib import Path

import torch


# Root paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = PROJECT_ROOT
TRAIN_IMAGES_DIR = DATA_ROOT / "images"
FUTURE_PASS_DIR = DATA_ROOT / "future_pass_images"
ANNOTATIONS_FILE = DATA_ROOT / "annotations.json"

# Train/val split by base image ID to avoid patch leakage
IMAGE_SPLIT_METADATA = DATA_ROOT / "splits.json"

# Output directories
OUTPUT_DIR = PROJECT_ROOT / "outputs"
MODELS_DIR = PROJECT_ROOT / "models"
TRAINING_HISTORY_FILE = OUTPUT_DIR / "training_history.json"
TRAINING_SUMMARY_FILE = OUTPUT_DIR / "training_summary.json"
CALIBRATED_THRESHOLDS_FILE = OUTPUT_DIR / "calibrated_thresholds.json"


# Device (CUDA > MPS > CPU)
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")


# Training hyperparameters
RANDOM_SEED = 1337
TRAIN_VAL_SPLIT = 0.9  # proportion of data used for training

NUM_CLASSES = 3  # background + small-vehicle + large-vehicle
CLASS_ID_MAP = {
    1: 1,  # small-vehicle -> index 1
    2: 2,  # large-vehicle -> index 2
}

TRAIN_IMAGE_SIZE = 896
VAL_IMAGE_SIZE = 896
INFER_IMAGE_SIZE = 896

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

BATCH_SIZE = 2
NUM_EPOCHS = 100
WARMUP_EPOCHS = 5
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 5e-4
COSINE_FINAL_LR = 1e-6
EMA_DECAY = 0.999
GRAD_ACCUM_STEPS = 1

GRAD_CLIP_NORM = 2.0

NUM_WORKERS = 4
PREFETCH_FACTOR = 2
PIN_MEMORY = True


# Inference
SCORE_THRESHOLD_SMALL = 0.2
SCORE_THRESHOLD_LARGE = 0.3
SCORE_THRESHOLDS = {1: SCORE_THRESHOLD_SMALL, 2: SCORE_THRESHOLD_LARGE}
MAX_DETECTIONS_PER_IMAGE = 200

THRESHOLD_SEARCH_GRID = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45]


# RPN settings tuned for small objects
RPN_PRE_NMS_TOP_N_TRAIN = 1000
RPN_PRE_NMS_TOP_N_TEST = 1000
RPN_POST_NMS_TOP_N_TRAIN = 300
RPN_POST_NMS_TOP_N_TEST = 300


def ensure_directories() -> None:
    """Ensure that output and model directories exist."""

    OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
    MODELS_DIR.mkdir(exist_ok=True, parents=True)


