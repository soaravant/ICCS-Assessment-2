# xView Vehicle Detection - ICCS Assessment 2

A PyTorch-based deep learning system for detecting and counting small and large vehicles in xView satellite imagery using Faster R-CNN.

## Features

- **Real-time Training Monitor**: Live visualization of training progress with matplotlib
- **Multi-device Support**: Optimized for both Apple Silicon (MPS) and CUDA GPUs
- **Flexible Training**: Multiple configurations for different hardware setups
- **Vehicle Counting**: Accurate detection and counting of small and large vehicles
- **Data Augmentation**: Comprehensive augmentation pipeline for satellite imagery

## Quick Start

### 1. Environment Setup

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Training

#### For Apple Silicon (M3 Max) - Fast Mode
```bash
python -m src.train \
  --epochs 6 \
  --batch_size 4 \
  --lr 3e-4 \
  --num_workers 4 \
  --prefetch_factor 1 \
  --pin_memory \
  --accumulate 2 \
  --img_size 640 \
  --trainable_layers 2 \
  --fast_aug \
  --val_every 2 \
  --train_ratio 0.9 \
  --max_steps 80 \
  --fast_mode
```

#### For CUDA GPUs (RTX 5060) - High Performance
```bash
./train_cuda.sh
```

### 3. Real-time Monitoring

In a separate terminal:
```bash
./start_monitor.sh
```

### 4. Inference

```bash
# Run inference on unlabeled images
python -m src.inference --model_path models/xview_vehicle_detector.pt --use_ema --tta

# Calibrate thresholds for better counting accuracy
python -m src.eval_thresholds --model_path models/xview_vehicle_detector.pt --use_ema
```

## Project Structure

```
├── src/
│   ├── config.py          # Configuration and hyperparameters
│   ├── dataset.py         # xView dataset loading and preprocessing
│   ├── engine.py          # Training and evaluation loops
│   ├── model.py           # Faster R-CNN model definitions
│   ├── train.py           # Main training script
│   ├── inference.py       # Inference on unlabeled images
│   ├── eval_thresholds.py # Threshold calibration
│   ├── monitor.py         # Real-time training visualization
│   ├── transforms.py      # Data augmentation pipelines
│   └── utils.py           # Utility functions
├── images/                # Training images (989 TIFF files)
├── future_pass_images/    # Unlabeled test images (20 TIFF files)
├── annotations.json       # COCO format annotations
├── outputs/               # Training outputs and metrics
├── models/                # Saved model checkpoints
├── train_cuda.sh          # CUDA-optimized training script
├── start_monitor.sh       # Real-time monitoring launcher
└── requirements.txt       # Python dependencies
```

## Hardware Optimizations

### Apple Silicon (M3 Max)
- Uses MobileNetV3-320-FPN for faster inference
- MPS (Metal Performance Shaders) acceleration
- Reduced RPN proposals for speed
- Memory-efficient training with cache clearing

### CUDA GPUs (RTX 5060)
- Mixed precision training with CUDA AMP
- Higher batch sizes and worker counts
- Full ResNet-50-FPN backbone
- Optimized for 8GB+ VRAM

## Training Configurations

| Configuration | Device | Batch Size | Epochs | Time | Quality |
|---------------|--------|------------|--------|------|---------|
| Fast Mode | MPS | 4 | 6 | ~2h | Good |
| Balanced | MPS | 4 | 8 | ~4h | Better |
| High Quality | CUDA | 8 | 8 | ~3h | Best |

## Output Files

- `models/xview_vehicle_detector.pt`: Trained model checkpoint
- `outputs/training_history.json`: Detailed training metrics
- `outputs/training_summary.json`: Compact training summary
- `outputs/calibrated_thresholds.json`: Optimized detection thresholds
- `outputs/inference_results.json`: Vehicle counts for test images

## Performance Metrics

The system tracks:
- **Detection Metrics**: IoU, Precision, Recall
- **Counting Accuracy**: MAE, RMSE for small/large vehicles
- **Training Speed**: Images per second, epoch duration
- **Memory Usage**: GPU/CPU utilization

## Requirements

- Python 3.9+
- PyTorch 2.0+
- CUDA 11.8+ (for GPU training)
- macOS 12+ (for MPS support)
- 16GB+ RAM recommended
- 8GB+ VRAM for CUDA training

## License

This project is part of ICCS Assessment 2.
