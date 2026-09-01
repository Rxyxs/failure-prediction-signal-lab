"""Features de dominio temporal para un segmento de senal acustica.

Incluye estadisticos descriptivos globales, cuartiles rodantes calculados
sobre sub-ventanas del segmento, momentos de orden superior (kurtosis,
skewness), desviacion absoluta mediana (MAD) y pendiente de tendencia lineal.

Autor: Pablo Reyes
"""
from __future__ import annotations

import numpy as np
from scipy import stats


def extract_time_domain_features(signal: np.ndarray, n_rolling_windows: int = 10) -> dict:
    """Calcula un diccionario de features de dominio temporal para ``signal``."""
    signal = signal.astype(np.float64)
    n = len(signal)

    features: dict[str, float] = {
        "td_mean": float(np.mean(signal)),
        "td_std": float(np.std(signal)),
        "td_var": float(np.var(signal)),
        "td_min": float(np.min(signal)),
        "td_max": float(np.max(signal)),
        "td_abs_mean": float(np.mean(np.abs(signal))),
        "td_abs_max": float(np.max(np.abs(signal))),
        "td_kurtosis": float(stats.kurtosis(signal)),
        "td_skewness": float(stats.skew(signal)),
        "td_mad": float(stats.median_abs_deviation(signal)),
        "td_median": float(np.median(signal)),
        "td_q01": float(np.quantile(signal, 0.01)),
        "td_q05": float(np.quantile(signal, 0.05)),
        "td_q25": float(np.quantile(signal, 0.25)),
        "td_q75": float(np.quantile(signal, 0.75)),
        "td_q95": float(np.quantile(signal, 0.95)),
        "td_q99": float(np.quantile(signal, 0.99)),
        "td_iqr": float(np.quantile(signal, 0.75) - np.quantile(signal, 0.25)),
    }

    # Tendencia lineal (pendiente de un ajuste de mínimos cuadrados).
    x = np.arange(n)
    slope, intercept, *_ = stats.linregress(x, signal)
    features["td_trend_slope"] = float(slope)
    features["td_trend_intercept"] = float(intercept)

    # Cuartiles rodantes: se divide la senal en sub-ventanas y se registran
    # estadisticos de cada una, capturando la evolucion temporal dentro del
    # segmento (util para detectar precursores que crecen hacia el final).
    window_size = max(1, n // n_rolling_windows)
    rolling_means = []
    rolling_stds = []
    rolling_q75 = []
    rolling_q25 = []
    for i in range(n_rolling_windows):
        start = i * window_size
        end = n if i == n_rolling_windows - 1 else start + window_size
        chunk = signal[start:end]
        if chunk.size == 0:
            continue
        rolling_means.append(np.mean(chunk))
        rolling_stds.append(np.std(chunk))
        rolling_q75.append(np.quantile(chunk, 0.75))
        rolling_q25.append(np.quantile(chunk, 0.25))

    features["td_rolling_mean_avg"] = float(np.mean(rolling_means))
    features["td_rolling_mean_std"] = float(np.std(rolling_means))
    features["td_rolling_std_avg"] = float(np.mean(rolling_stds))
    features["td_rolling_std_std"] = float(np.std(rolling_stds))
    features["td_rolling_q75_last_minus_first"] = float(rolling_q75[-1] - rolling_q75[0])
    features["td_rolling_q25_last_minus_first"] = float(rolling_q25[-1] - rolling_q25[0])

    return features
