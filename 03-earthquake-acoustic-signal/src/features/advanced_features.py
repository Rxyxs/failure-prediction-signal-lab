"""Features avanzadas de senal: entropia, autocorrelacion y conteo de picos.

Estas features complementan las estadisticas de dominio temporal y de
frecuencia ya existentes, aportando descriptores no lineales de la dinamica
de la senal que son particularmente relevantes para precursores sismicos:

- Entropia de Shannon del histograma de amplitudes: mide cuan "desordenada"
  o impredecible es la distribucion de valores de la senal. Los precursores
  sismicos tienden a reducir la entropia local al introducir estructura
  (pulsos repetidos) sobre el ruido de fondo.
- Autocorrelacion a distintos lags: captura periodicidad y memoria de corto
  plazo en la senal, util para distinguir ruido blanco de patrones
  cuasi-periodicos asociados a microsismos.
- Conteo de picos por encima de umbrales (en multiplos de la desviacion
  estandar): cuantifica la frecuencia de eventos impulsivos de gran
  amplitud, que es la senal mas directa de actividad precursora.

Autor: Pablo Reyes
"""
from __future__ import annotations

import numpy as np
from scipy.signal import find_peaks


def extract_entropy_features(signal: np.ndarray, n_bins: int = 50) -> dict:
    """Calcula la entropia de Shannon de la distribucion de amplitudes de ``signal``."""
    signal = signal.astype(np.float64)
    hist, _ = np.histogram(signal, bins=n_bins, density=False)
    probs = hist / (hist.sum() + 1e-12)
    probs = probs[probs > 0]
    shannon_entropy = float(-np.sum(probs * np.log2(probs)))

    # Entropia aproximada sobre una version discretizada en signos (+/-) de la
    # primera diferencia: mide la impredecibilidad de la direccion de cambio.
    diffs = np.diff(signal)
    signs = (diffs > 0).astype(int)
    if signs.size > 0:
        p1 = float(np.mean(signs))
        p0 = 1.0 - p1
        bits = [p for p in (p0, p1) if p > 0]
        sign_entropy = float(-sum(p * np.log2(p) for p in bits))
    else:
        sign_entropy = 0.0

    return {
        "entropy_shannon": shannon_entropy,
        "entropy_sign_diff": sign_entropy,
    }


def extract_autocorrelation_features(signal: np.ndarray, lags: tuple[int, ...] = (1, 5, 10, 50, 100)) -> dict:
    """Calcula el coeficiente de autocorrelacion de ``signal`` para varios lags."""
    signal = signal.astype(np.float64)
    n = len(signal)
    centered = signal - signal.mean()
    denom = float(np.sum(centered ** 2)) + 1e-12

    features: dict[str, float] = {}
    for lag in lags:
        if lag >= n:
            features[f"autocorr_lag_{lag}"] = 0.0
            continue
        numerator = float(np.sum(centered[: n - lag] * centered[lag:]))
        features[f"autocorr_lag_{lag}"] = numerator / denom
    return features


def extract_peak_count_features(
    signal: np.ndarray, thresholds_in_std: tuple[float, ...] = (2.0, 4.0, 6.0, 8.0)
) -> dict:
    """Cuenta picos de ``signal`` por encima de multiplos de su desviacion estandar."""
    signal = signal.astype(np.float64)
    std = float(np.std(signal)) + 1e-12
    abs_signal = np.abs(signal - np.mean(signal))

    features: dict[str, float] = {}
    for thr in thresholds_in_std:
        height = thr * std
        peaks, _ = find_peaks(abs_signal, height=height)
        key = str(thr).replace(".", "_")
        features[f"peak_count_gt_{key}std"] = float(len(peaks))

    # Distancia promedio entre picos significativos (>= 4 std), como proxy de
    # la frecuencia de recurrencia de micro-eventos.
    peaks_main, _ = find_peaks(abs_signal, height=4.0 * std)
    if len(peaks_main) > 1:
        features["peak_mean_gap"] = float(np.mean(np.diff(peaks_main)))
    else:
        features["peak_mean_gap"] = float(len(signal))

    return features
