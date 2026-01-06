# Inference Guide

This guide explains how to use the seismic event classification model for inference.

## Prerequisites

1. Activate the conda environment:
   ```bash
   conda activate ml_project
   ```

2. Ensure you have the required artifacts:
   - `models/scaler.pkl` - Feature scaler (tracked in git)
   - `models/best_seismic-v1.ckpt` - Trained model checkpoint (NOT tracked, obtain from training)

## Quick Start

```bash
# Run inference on a sample file
python scripts/predict.py --input data/sample_input.json

# With custom threshold
python scripts/predict.py --input data.json --threshold 0.6

# Save output to file
python scripts/predict.py --input data.json --output results.json
```

## Input Format

The input JSON file must contain three acceleration arrays:

```json
{
  "AccV": [0.001, 0.002, ...],
  "AccH1": [0.001, 0.002, ...],
  "AccH2": [0.001, 0.002, ...],
  "dt": 0.01
}
```

| Field | Description | Required |
|-------|-------------|----------|
| `AccV` | Vertical acceleration component (array) | Yes |
| `AccH1` | Horizontal 1 acceleration component (array) | Yes |
| `AccH2` | Horizontal 2 acceleration component (array) | Yes |
| `dt` | Time step in seconds (default: 0.01) | No |

## Output Format

```json
{
  "labels": ["1 Stiker Slip (SS)", "4-6", "400-600"],
  "scores": {
    "1 Stiker Slip (SS)": 0.85,
    "2 Normal-Oblique (SO)": 0.12,
    "3 Reverse-Oblique (RO)": 0.23,
    "4-6": 0.92,
    "6-8": 0.18,
    "0-200": 0.05,
    "200-400": 0.32,
    "400-600": 0.78,
    "600-": 0.15
  },
  "threshold": 0.5,
  "raw_predictions": {
    "1 Stiker Slip (SS)": true,
    "2 Normal-Oblique (SO)": false,
    ...
  }
}
```

| Field | Description |
|-------|-------------|
| `labels` | List of predicted labels (above threshold) |
| `scores` | Raw probability scores for each label |
| `threshold` | Classification threshold used |
| `raw_predictions` | Boolean predictions for each label |

## Label Meanings

### Fault Type
- **1 Stiker Slip (SS)**: Strike-slip fault mechanism
- **2 Normal-Oblique (SO)**: Normal-oblique fault mechanism
- **3 Reverse-Oblique (RO)**: Reverse-oblique fault mechanism

### Magnitude
- **4-6**: Earthquake magnitude between 4 and 6
- **6-8**: Earthquake magnitude between 6 and 8

### Vs30 (Soil Classification)
- **0-200**: Very soft soil (Vs30 < 200 m/s)
- **200-400**: Soft soil (Vs30 200-400 m/s)
- **400-600**: Stiff soil (Vs30 400-600 m/s)
- **600-**: Rock (Vs30 > 600 m/s)

## CLI Options

```
usage: predict.py [-h] --input INPUT [--model MODEL] [--scaler SCALER]
                  [--threshold THRESHOLD] [--output OUTPUT]

options:
  --input INPUT       Path to input JSON file (required)
  --model MODEL       Path to model checkpoint (default: models/best_seismic-v1.ckpt)
  --scaler SCALER     Path to scaler.pkl (default: models/scaler.pkl)
  --threshold FLOAT   Classification threshold (default: 0.5)
  --output OUTPUT     Output JSON file (default: stdout)
```

## Troubleshooting

### Model checkpoint not found
The `.ckpt` files are not tracked in git to keep the repository size small.
You need to either:
1. Train a model using notebook `07_Redes_model.ipynb`
2. Obtain a pre-trained checkpoint from a team member

### Feature dimension mismatch
If you see a warning about feature dimension mismatch, it means the model
was trained with PCA-reduced features. The inference script handles this
automatically, but results may vary.

### Import errors
Ensure you have activated the conda environment:
```bash
conda activate ml_project
```

## Programmatic Usage

```python
from pathlib import Path
import json

# Add project root to path
import sys
sys.path.insert(0, str(Path(__file__).parents[1]))

from scripts.predict import load_artifacts, predict

# Load artifacts
model, scaler = load_artifacts()

# Prepare input data
data = {
    "AccV": [...],
    "AccH1": [...],
    "AccH2": [...],
    "dt": 0.01,
}

# Run prediction
result = predict(data, model, scaler, threshold=0.5)
print(json.dumps(result, indent=2))
```
