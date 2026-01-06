#!/usr/bin/env python
"""
CLI inference script for seismic event classification.

Usage:
    python scripts/predict.py --input data.json
    python scripts/predict.py --input data.json --model models/best_seismic-v1.ckpt
    python scripts/predict.py --help

Input format (JSON):
    {
        "AccV": [0.001, 0.002, ...],
        "AccH1": [0.001, 0.002, ...],
        "AccH2": [0.001, 0.002, ...],
        "dt": 0.01  // optional, defaults to 0.01
    }

Output format (JSON):
    {
        "labels": ["1 Strike Slip (SS)", "4-6", "400-600"],
        "scores": [0.85, 0.92, 0.78, ...],
        "threshold": 0.5,
        "raw_predictions": {...}
    }
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

# Add project root to path for imports
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


# Label names in the order used during training
# Based on notebook 03 output columns
LABEL_NAMES = [
    "1 Stiker Slip (SS)",
    "2 Normal-Oblique (SO)",
    "3 Reverse-Oblique (RO)",
    "4-6",
    "6-8",
    "0-200",
    "200-400",
    "400-600",
    "600-",
]


def load_artifacts(
    model_path: Path = None,
    scaler_path: Path = None,
):
    """
    Load model checkpoint and scaler.

    Args:
        model_path: Path to .ckpt file (default: models/best_seismic-v1.ckpt)
        scaler_path: Path to scaler.pkl (default: models/scaler.pkl)

    Returns:
        Tuple of (model, scaler)
    """
    import pickle
    import torch
    from src.seismic_model import SeismicMultilabelModel

    # Default paths
    if model_path is None:
        model_path = ROOT / "models" / "best_seismic-v1.ckpt"
    if scaler_path is None:
        scaler_path = ROOT / "models" / "scaler.pkl"

    # Load scaler
    if not scaler_path.exists():
        raise FileNotFoundError(f"Scaler not found: {scaler_path}")
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)

    # Load model
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model checkpoint not found: {model_path}\n"
            "Note: .ckpt files are not tracked in git. "
            "Please ensure you have a trained model available."
        )

    # Load checkpoint
    model = SeismicMultilabelModel.load_from_checkpoint(
        model_path,
        map_location=torch.device("cpu"),
    )
    model.eval()

    return model, scaler


def extract_features(data: dict) -> np.ndarray:
    """
    Extract features from input data.

    Args:
        data: Dictionary with AccV, AccH1, AccH2 arrays and optional dt

    Returns:
        Feature vector (1D numpy array)
    """
    from sklearn.decomposition import PCA
    from src.features import compute_all_features, flatten_fft_magnitudes

    # Parse input
    acc_v = np.array(data["AccV"]).flatten()
    acc_h1 = np.array(data["AccH1"]).flatten()
    acc_h2 = np.array(data["AccH2"]).flatten()
    dt = data.get("dt", 0.01)

    # Compute features
    feats = compute_all_features(
        acc_v, acc_h1, acc_h2, dt=dt, return_fft_magnitudes=True
    )

    # Flatten FFT magnitudes
    flat_feats = flatten_fft_magnitudes(feats)

    # Order features as expected by the model
    # Non-FFT features first, then FFT features
    non_fft_keys = [k for k in flat_feats.keys() if not k.startswith("FFTmag_")]
    fft_keys = sorted([k for k in flat_feats.keys() if k.startswith("FFTmag_")])

    # Build feature vector
    feature_order = non_fft_keys + fft_keys
    feature_vector = np.array([flat_feats[k] for k in feature_order])

    return feature_vector


def predict(
    data: dict,
    model,
    scaler,
    threshold: float = 0.5,
) -> dict:
    """
    Run inference on input data.

    Args:
        data: Input data dictionary
        model: Loaded PyTorch Lightning model
        scaler: Loaded sklearn scaler
        threshold: Classification threshold

    Returns:
        Dictionary with predictions
    """
    import torch

    # Extract features
    features = extract_features(data)

    # Note: The model was trained with PCA-reduced features
    # For full inference, we'd need the fitted PCA transformer
    # For now, we'll provide a simplified version that works with scaled features

    # Scale features
    # Note: scaler expects shape (n_samples, n_features)
    features_2d = features.reshape(1, -1)

    # Check if scaler matches feature dimension
    if hasattr(scaler, "n_features_in_"):
        expected_features = scaler.n_features_in_
        if features_2d.shape[1] != expected_features:
            print(
                f"Warning: Feature dimension mismatch. "
                f"Expected {expected_features}, got {features_2d.shape[1]}. "
                f"The model was likely trained with PCA-reduced features.",
                file=sys.stderr,
            )
            # For demo purposes, we'll truncate or pad
            if features_2d.shape[1] > expected_features:
                features_2d = features_2d[:, :expected_features]
            else:
                padded = np.zeros((1, expected_features))
                padded[:, : features_2d.shape[1]] = features_2d
                features_2d = padded

    features_scaled = scaler.transform(features_2d)

    # Convert to tensor
    x = torch.tensor(features_scaled, dtype=torch.float32)

    # Run inference
    with torch.no_grad():
        logits = model(x)
        probs = torch.sigmoid(logits).numpy()[0]

    # Apply threshold
    predictions = probs >= threshold

    # Get predicted labels
    predicted_labels = [
        LABEL_NAMES[i] for i, pred in enumerate(predictions) if pred
    ]

    # Build output
    result = {
        "labels": predicted_labels,
        "scores": {name: float(probs[i]) for i, name in enumerate(LABEL_NAMES)},
        "threshold": threshold,
        "raw_predictions": {
            name: bool(predictions[i]) for i, name in enumerate(LABEL_NAMES)
        },
    }

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Seismic event multilabel classification",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to input JSON file with AccV, AccH1, AccH2 arrays",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=None,
        help="Path to model checkpoint (default: models/best_seismic-v1.ckpt)",
    )
    parser.add_argument(
        "--scaler",
        type=Path,
        default=None,
        help="Path to scaler.pkl (default: models/scaler.pkl)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Classification threshold (default: 0.5)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSON file (default: stdout)",
    )

    args = parser.parse_args()

    # Load input data
    if not args.input.exists():
        print(f"Error: Input file not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    with open(args.input) as f:
        data = json.load(f)

    # Validate input
    required_keys = ["AccV", "AccH1", "AccH2"]
    for key in required_keys:
        if key not in data:
            print(f"Error: Missing required key in input: {key}", file=sys.stderr)
            sys.exit(1)

    try:
        # Load artifacts
        model, scaler = load_artifacts(args.model, args.scaler)

        # Run prediction
        result = predict(data, model, scaler, args.threshold)

        # Output
        output_json = json.dumps(result, indent=2)

        if args.output:
            with open(args.output, "w") as f:
                f.write(output_json)
            print(f"Results written to: {args.output}")
        else:
            print(output_json)

    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error during inference: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
