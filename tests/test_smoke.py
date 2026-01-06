"""
Smoke tests for core functionality.

These tests are designed to run quickly (<5s total) and verify
that the main components work correctly with synthetic data.
"""

import numpy as np
import pytest


class TestFeatureExtraction:
    """Tests for feature extraction functions."""

    @pytest.fixture
    def sine_wave(self):
        """Generate a simple sine wave for testing."""
        dt = 0.01  # 100 Hz
        t = np.arange(0, 1, dt)  # 1 second
        freq = 10  # Hz
        signal = np.sin(2 * np.pi * freq * t)
        return signal, dt, freq

    def test_extract_time_features(self, sine_wave):
        """Test time-domain feature extraction."""
        from src.features import extract_time_features

        signal, dt, _ = sine_wave
        feats = extract_time_features(signal, dt, "test")

        assert "max_test" in feats
        assert "rms_test" in feats
        # Sine wave max should be close to 1.0
        assert abs(feats["max_test"] - 1.0) < 0.1
        # Sine wave RMS should be close to 1/sqrt(2)
        assert abs(feats["rms_test"] - 0.707) < 0.1

    def test_zero_crossing_rate(self, sine_wave):
        """Test zero-crossing rate calculation."""
        from src.features import zero_crossing_rate

        signal, dt, freq = sine_wave
        zcr = zero_crossing_rate(signal, dt)

        # 10 Hz sine wave should have ~20 zero crossings per second
        # (2 crossings per cycle)
        expected_zcr = freq * 2
        assert abs(zcr - expected_zcr) < 5  # Allow some tolerance

    def test_fft_features(self, sine_wave):
        """Test FFT-based feature extraction."""
        from src.features import fft_features

        signal, dt, freq = sine_wave
        feats = fft_features(signal, dt, "test", return_magnitude=True)

        assert "dom_freq_test" in feats
        assert "centroid_test" in feats
        assert "bandwidth_test" in feats
        assert "spec_ratio_test" in feats
        assert "FFTmag_test" in feats

        # Dominant frequency should be close to 10 Hz
        assert abs(feats["dom_freq_test"] - freq) < 2

    def test_compute_all_features(self):
        """Test unified feature computation."""
        from src.features import compute_all_features

        dt = 0.01
        n_samples = 100
        acc_v = np.random.randn(n_samples) * 0.001
        acc_h1 = np.random.randn(n_samples) * 0.001
        acc_h2 = np.random.randn(n_samples) * 0.001

        feats = compute_all_features(acc_v, acc_h1, acc_h2, dt=dt)

        # Check that all expected keys are present
        expected_keys = [
            "max_V", "rms_V", "max_H1", "rms_H1", "max_H2", "rms_H2",
            "duration", "zcr_V",
            "dom_freq_V", "centroid_V", "bandwidth_V", "spec_ratio_V",
            "dom_freq_H1", "centroid_H1", "bandwidth_H1", "spec_ratio_H1",
            "dom_freq_H2", "centroid_H2", "bandwidth_H2", "spec_ratio_H2",
        ]
        for key in expected_keys:
            assert key in feats, f"Missing feature: {key}"

    def test_flatten_fft_magnitudes(self):
        """Test FFT magnitude flattening."""
        from src.features import fft_features, flatten_fft_magnitudes

        signal = np.random.randn(100)
        feats = fft_features(signal, 0.01, "V", n_fft=64, return_magnitude=True)

        # Add dummy features for other components
        feats["FFTmag_H1"] = feats["FFTmag_V"]
        feats["FFTmag_H2"] = feats["FFTmag_V"]

        flat = flatten_fft_magnitudes(feats)

        # Should have individual columns for each FFT bin
        assert "FFTmag_V_0" in flat
        assert "FFTmag_H1_0" in flat
        assert "FFTmag_H2_0" in flat
        # Should not have array features
        assert "FFTmag_V" not in flat


class TestModelInstantiation:
    """Tests for model instantiation."""

    def test_model_creation(self):
        """Test that model can be instantiated."""
        from src.seismic_model import SeismicMultilabelModel

        model = SeismicMultilabelModel(
            input_dim=82,
            hidden_units=(64, 32),
            dropout_rate=0.2,
            lr=0.001,
            l2=0.0001,
            num_classes=9,
        )

        assert model is not None
        assert hasattr(model, "forward")
        assert hasattr(model, "training_step")
        assert hasattr(model, "validation_step")

    def test_model_forward_pass(self):
        """Test that model can perform a forward pass."""
        import torch
        from src.seismic_model import SeismicMultilabelModel

        input_dim = 82
        batch_size = 4
        num_classes = 9

        model = SeismicMultilabelModel(
            input_dim=input_dim,
            hidden_units=(64,),
            num_classes=num_classes,
        )
        model.eval()

        # Create dummy input
        x = torch.randn(batch_size, input_dim)

        # Forward pass
        with torch.no_grad():
            output = model(x)

        assert output.shape == (batch_size, num_classes)

    def test_model_save_hyperparameters(self):
        """Test that model saves hyperparameters correctly."""
        from src.seismic_model import SeismicMultilabelModel

        model = SeismicMultilabelModel(
            input_dim=82,
            hidden_units=(128, 64),
            dropout_rate=0.3,
            lr=0.01,
            l2=0.001,
            num_classes=9,
        )

        assert model.hparams.input_dim == 82
        assert model.hparams.hidden_units == (128, 64)
        assert model.hparams.dropout_rate == 0.3
        assert model.hparams.lr == 0.01
        assert model.hparams.num_classes == 9
