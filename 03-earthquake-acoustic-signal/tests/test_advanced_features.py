"""Pruebas de las features avanzadas: entropia, autocorrelacion y conteo de picos.

Autor: Pablo Reyes
"""
from __future__ import annotations

import numpy as np

from src.features.advanced_features import (
    extract_autocorrelation_features,
    extract_entropy_features,
    extract_peak_count_features,
)


def _make_test_signal(n: int = 20_000) -> np.ndarray:
    rng = np.random.default_rng(0)
    t = np.arange(n)
    return (np.sin(2 * np.pi * t / 50.0) * 5.0 + rng.normal(0, 1.0, n)).astype(np.float64)


def test_entropy_features_are_finite_and_positive():
    signal = _make_test_signal()
    features = extract_entropy_features(signal)
    assert features["entropy_shannon"] > 0
    assert 0.0 <= features["entropy_sign_diff"] <= 1.0
    for value in features.values():
        assert np.isfinite(value)


def test_entropy_of_constant_signal_is_zero():
    signal = np.ones(1000)
    features = extract_entropy_features(signal)
    assert abs(features["entropy_shannon"]) < 1e-9


def test_autocorrelation_lag_zero_like_behaviour():
    signal = _make_test_signal()
    features = extract_autocorrelation_features(signal, lags=(1, 5, 50))
    assert set(features.keys()) == {"autocorr_lag_1", "autocorr_lag_5", "autocorr_lag_50"}
    for value in features.values():
        assert -1.0001 <= value <= 1.0001


def test_autocorrelation_periodic_signal_has_high_correlation_at_period_lag():
    n = 5000
    t = np.arange(n)
    periodic_signal = np.sin(2 * np.pi * t / 40.0)
    features = extract_autocorrelation_features(periodic_signal, lags=(40,))
    assert features["autocorr_lag_40"] > 0.9


def test_autocorrelation_handles_lag_larger_than_signal():
    signal = np.arange(10, dtype=np.float64)
    features = extract_autocorrelation_features(signal, lags=(100,))
    assert features["autocorr_lag_100"] == 0.0


def test_peak_count_features_detect_injected_spikes():
    rng = np.random.default_rng(1)
    signal = rng.normal(0, 1.0, 5000)
    spike_positions = [500, 1500, 2500, 3500, 4500]
    for pos in spike_positions:
        signal[pos] = 50.0

    features = extract_peak_count_features(signal, thresholds_in_std=(2.0, 4.0))
    assert features["peak_count_gt_2_0std"] >= len(spike_positions)
    assert features["peak_count_gt_4_0std"] >= len(spike_positions)
    assert features["peak_mean_gap"] > 0


def test_peak_count_features_no_peaks_in_flat_signal():
    signal = np.zeros(1000)
    features = extract_peak_count_features(signal, thresholds_in_std=(2.0,))
    assert features["peak_count_gt_2_0std"] == 0.0
