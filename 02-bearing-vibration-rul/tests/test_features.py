import numpy as np

from src.features import extract_features


def test_pure_sine_dominant_frequency_matches():
    fs = 20_000
    t = np.arange(20_480) / fs
    freq = 1000.0
    signal = np.sin(2 * np.pi * freq * t)
    feats = extract_features(signal)
    assert abs(feats["dominant_frequency_hz"] - freq) < 5.0


def test_constant_signal_has_zero_std_and_crest_factor_defined():
    signal = np.ones(20_480) * 0.5
    feats = extract_features(signal)
    assert feats["std"] < 1e-9
    assert np.isfinite(feats["crest_factor"])


def test_rms_matches_hand_computation():
    signal = np.array([1.0, -1.0, 1.0, -1.0] * 5120)
    feats = extract_features(signal)
    assert abs(feats["rms"] - 1.0) < 1e-9
