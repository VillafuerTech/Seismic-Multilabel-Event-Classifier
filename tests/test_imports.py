"""
Test that all key modules can be imported successfully.

These tests verify that:
1. All dependencies are installed correctly
2. Module structure is correct
3. No import-time errors occur
"""

import pytest


def test_import_numpy():
    """Test numpy import."""
    import numpy as np
    assert hasattr(np, "array")


def test_import_pandas():
    """Test pandas import."""
    import pandas as pd
    assert hasattr(pd, "DataFrame")


def test_import_scipy():
    """Test scipy.fft import."""
    from scipy.fft import fft, fftfreq
    assert callable(fft)
    assert callable(fftfreq)


def test_import_sklearn():
    """Test scikit-learn import."""
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    assert callable(StandardScaler)
    assert callable(PCA)


def test_import_torch():
    """Test PyTorch import."""
    import torch
    assert hasattr(torch, "tensor")


def test_import_lightning():
    """Test PyTorch Lightning import."""
    import pytorch_lightning as pl
    assert hasattr(pl, "LightningModule")


def test_import_src_features():
    """Test src.features module import."""
    from src.features import (
        extract_time_features,
        zero_crossing_rate,
        fft_features,
        compute_all_features,
    )
    assert callable(extract_time_features)
    assert callable(zero_crossing_rate)
    assert callable(fft_features)
    assert callable(compute_all_features)


def test_import_src_model():
    """Test src.seismic_model import."""
    from src.seismic_model import SeismicMultilabelModel
    assert callable(SeismicMultilabelModel)


def test_import_src_package():
    """Test src package-level imports."""
    from src import (
        extract_time_features,
        zero_crossing_rate,
        fft_features,
        compute_all_features,
        SeismicMultilabelModel,
    )
    assert callable(extract_time_features)
    assert callable(SeismicMultilabelModel)
