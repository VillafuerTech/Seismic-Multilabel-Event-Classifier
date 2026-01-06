"""
Feature extraction functions for seismic signal classification.

This module contains the core FFT-based feature extraction logic
used throughout the seismic event classification pipeline.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from scipy.fft import fft, fftfreq

# Repository root for consistent path resolution
ROOT = Path(__file__).resolve().parents[1]

# Default FFT parameters
DEFAULT_N_FFT = 512
DEFAULT_BAND_SPLIT = 5.0  # Hz
DEFAULT_DT = 0.01  # seconds (100 Hz sampling)


def extract_time_features(sig: np.ndarray, dt: float, prefix: str) -> Dict[str, float]:
    """
    Extract time-domain features from a signal.

    Args:
        sig: 1D numpy array of signal values
        dt: Time step between samples (seconds)
        prefix: String prefix for feature names (e.g., 'V', 'H1', 'H2')

    Returns:
        Dictionary with keys:
        - max_{prefix}: Maximum absolute value
        - rms_{prefix}: Root mean square value
    """
    feats = {
        f"max_{prefix}": float(np.max(np.abs(sig))),
        f"rms_{prefix}": float(np.sqrt(np.mean(sig ** 2))),
    }
    return feats


def zero_crossing_rate(sig: np.ndarray, dt: float) -> float:
    """
    Calculate the zero-crossing rate of a signal.

    Args:
        sig: 1D numpy array of signal values
        dt: Time step between samples (seconds)

    Returns:
        Zero-crossing rate (crossings per second)
    """
    crossings = np.where(np.diff(np.signbit(sig)))[0]
    duration = len(sig) * dt
    return len(crossings) / duration if duration > 0 else 0.0


def fft_features(
    sig: np.ndarray,
    dt: float,
    prefix: str,
    n_fft: int = DEFAULT_N_FFT,
    band_split: float = DEFAULT_BAND_SPLIT,
    return_magnitude: bool = True,
) -> Dict[str, Union[float, np.ndarray]]:
    """
    Extract FFT-based spectral features from a signal.

    Args:
        sig: 1D numpy array of signal values
        dt: Time step between samples (seconds)
        prefix: String prefix for feature names (e.g., 'V', 'H1', 'H2')
        n_fft: Number of FFT points
        band_split: Frequency (Hz) to split low/high bands
        return_magnitude: If True, include FFT magnitude array in output

    Returns:
        Dictionary with keys:
        - dom_freq_{prefix}: Dominant frequency (Hz)
        - centroid_{prefix}: Spectral centroid (Hz)
        - bandwidth_{prefix}: Spectral bandwidth (Hz)
        - spec_ratio_{prefix}: High/low frequency energy ratio
        - FFTmag_{prefix}: (optional) FFT magnitude array
    """
    # Compute one-sided FFT magnitude
    mag = np.abs(fft(sig, n=n_fft))[: n_fft // 2 + 1]
    freqs = fftfreq(n_fft, d=dt)[: n_fft // 2 + 1]

    # Spectral features
    dom_freq = freqs[np.argmax(mag)]
    centroid = (freqs * mag).sum() / (mag.sum() + 1e-12)
    bandwidth = np.sqrt(((freqs - centroid) ** 2 * mag).sum() / (mag.sum() + 1e-12))

    # Band energy ratio
    low_energy = mag[freqs < band_split].sum()
    high_energy = mag[freqs >= band_split].sum()
    spec_ratio = high_energy / (low_energy + 1e-12)

    feats = {
        f"dom_freq_{prefix}": float(dom_freq),
        f"centroid_{prefix}": float(centroid),
        f"bandwidth_{prefix}": float(bandwidth),
        f"spec_ratio_{prefix}": float(spec_ratio),
    }

    if return_magnitude:
        feats[f"FFTmag_{prefix}"] = mag

    return feats


def compute_all_features(
    acc_v: np.ndarray,
    acc_h1: np.ndarray,
    acc_h2: np.ndarray,
    dt: float = DEFAULT_DT,
    n_fft: int = DEFAULT_N_FFT,
    return_fft_magnitudes: bool = False,
) -> Dict[str, Union[float, np.ndarray]]:
    """
    Compute all features for a single seismic record with three components.

    This is the main entry point for feature extraction, combining
    time-domain and frequency-domain features for all three components.

    Args:
        acc_v: Vertical acceleration component
        acc_h1: Horizontal 1 acceleration component
        acc_h2: Horizontal 2 acceleration component
        dt: Time step between samples (seconds)
        n_fft: Number of FFT points
        return_fft_magnitudes: If True, include FFT magnitude arrays

    Returns:
        Dictionary containing all extracted features:
        - Time domain: max_V, rms_V, max_H1, rms_H1, max_H2, rms_H2
        - Duration and ZCR: duration, zcr_V
        - FFT features for each component: dom_freq_*, centroid_*, bandwidth_*, spec_ratio_*
        - Optional: FFTmag_V, FFTmag_H1, FFTmag_H2
    """
    feats = {}

    # Component mapping
    components = {"V": acc_v, "H1": acc_h1, "H2": acc_h2}

    # Time-domain features for all components
    for comp_name, sig in components.items():
        feats.update(extract_time_features(sig, dt, comp_name))

    # Duration and zero-crossing rate (vertical component only)
    feats["duration"] = (len(acc_v) - 1) * dt
    feats["zcr_V"] = zero_crossing_rate(acc_v, dt)

    # FFT features for all components
    for comp_name, sig in components.items():
        feats.update(
            fft_features(
                sig, dt, comp_name, n_fft=n_fft, return_magnitude=return_fft_magnitudes
            )
        )

    return feats


def flatten_fft_magnitudes(
    features: Dict[str, Union[float, np.ndarray]],
    components: List[str] = ["V", "H1", "H2"],
) -> Dict[str, float]:
    """
    Flatten FFT magnitude arrays into individual columns.

    Args:
        features: Dictionary containing FFTmag_* arrays
        components: List of component names to process

    Returns:
        Dictionary with flattened features (FFTmag_V_0, FFTmag_V_1, etc.)
    """
    flat_feats = {}

    # Copy non-FFT features
    for key, value in features.items():
        if not key.startswith("FFTmag_"):
            flat_feats[key] = value

    # Flatten FFT magnitudes
    for comp in components:
        key = f"FFTmag_{comp}"
        if key in features:
            mag = features[key]
            for i, val in enumerate(mag):
                flat_feats[f"FFTmag_{comp}_{i}"] = float(val)

    return flat_feats
