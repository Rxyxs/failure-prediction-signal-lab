"""Pruebas de extraccion de features espectrales y temporales.

Autor: Pablo Reyes
"""
from __future__ import annotations

import numpy as np

from src.features.frequency_domain import (
    extract_fft_features,
    extract_hilbert_features,
    extract_spectrogram_features,
    extract_welch_psd_features,
)
from src.features.time_domain import extract_time_domain_features


def _make_test_signal(n: int = 20_000) -> np.ndarray:
    rng = np.random.default_rng(0)
    t = np.arange(n)
    return (np.sin(2 * np.pi * t / 50.0) * 5.0 + rng.normal(0, 1.0, n)).astype(np.float64)


def test_time_domain_features_keys_and_values():
    signal = _make_test_signal()
    features = extract_time_domain_features(signal, n_rolling_windows=5)

    expected_keys = {
        "td_mean", "td_std", "td_var", "td_min", "td_max", "td_kurtosis",
        "td_skewness", "td_mad", "td_trend_slope", "td_rolling_mean_avg",
    }
    assert expected_keys.issubset(features.keys())
    for value in features.values():
        assert np.isfinite(value)


def test_fft_features_dominant_frequency_detected():
    signal = _make_test_signal()
    features = extract_fft_features(signal)
    assert features["fft_energy"] > 0
    assert features["fft_dominant_freq"] > 0
    assert np.isclose(features["fft_dominant_freq"], 1.0 / 50.0, atol=0.01)


def test_welch_psd_features_bands_sum_reasonable():
    signal = _make_test_signal()
    features = extract_welch_psd_features(signal, nperseg=1024, n_bands=4)
    band_keys = [k for k in features if k.startswith("psd_band_")]
    assert len(band_keys) == 4
    assert features["psd_total_power"] > 0


def test_spectrogram_features_no_nan():
    signal = _make_test_signal()
    features = extract_spectrogram_features(signal, nperseg=512)
    for value in features.values():
        assert np.isfinite(value)


def test_hilbert_envelope_and_phase():
    signal = _make_test_signal()
    features = extract_hilbert_features(signal)
    assert features["hilbert_envelope_mean"] > 0
    assert features["hilbert_envelope_max"] >= features["hilbert_envelope_mean"]
    assert np.isfinite(features["hilbert_phase_std"])


def test_features_are_deterministic():
    signal = _make_test_signal()
    features_a = extract_time_domain_features(signal)
    features_b = extract_time_domain_features(signal)
    assert features_a == features_b
