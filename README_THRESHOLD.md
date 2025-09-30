# Threshold Calibration Utility

After training `src/train.py`, run the threshold sweep to minimize counting MAE on the validation split:

```bash
source .venv/bin/activate
python -m src.eval_thresholds --model_path models/xview_vehicle_detector.pt --use_ema
```

This writes tuned thresholds to `outputs/calibrated_thresholds.json`. Update `config.SCORE_THRESHOLDS` if you want to bake them into inference.酃


