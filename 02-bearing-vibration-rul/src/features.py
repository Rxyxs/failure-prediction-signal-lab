"""Feature engineering espectral: dominio temporal + dominio de frecuencia
(FFT) sobre cada snapshot de vibracion de 20.480 puntos."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew, entropy as scipy_entropy
from scipy.fft import rfft, rfftfreq

from src.ingest import SAMPLING_RATE_HZ, Snapshot, time_to_failure_fraction, time_to_failure_minutes


def _spectral_entropy(magnitude: np.ndarray) -> float:
    psd = magnitude**2
    psd_norm = psd / (psd.sum() + 1e-12)
    return float(scipy_entropy(psd_norm + 1e-12))


def extract_features(signal: np.ndarray) -> dict:
    n = len(signal)

    # --- dominio temporal ---
    rms = float(np.sqrt(np.mean(signal**2)))
    feats = {
        "mean": float(signal.mean()),
        "std": float(signal.std()),
        "rms": rms,
        "min": float(signal.min()),
        "max": float(signal.max()),
        "peak_to_peak": float(signal.max() - signal.min()),
        "crest_factor": float(np.max(np.abs(signal)) / (rms + 1e-9)),
        "kurtosis": float(kurtosis(signal)),
        "skew": float(skew(signal)),
        "q10": float(np.quantile(signal, 0.10)),
        "q25": float(np.quantile(signal, 0.25)),
        "q75": float(np.quantile(signal, 0.75)),
        "q90": float(np.quantile(signal, 0.90)),
        "iqr": float(np.quantile(signal, 0.75) - np.quantile(signal, 0.25)),
    }

    # cuartiles rodantes (ventana de 2000 puntos, ~100ms a 20kHz): capturan
    # si la dispersion del rodamiento crece de forma abrupta dentro del
    # propio snapshot, no solo entre snapshots.
    window = 2000
    n_windows = n // window
    if n_windows >= 2:
        rolled = signal[: n_windows * window].reshape(n_windows, window)
        rolling_std = rolled.std(axis=1)
        feats["rolling_std_mean"] = float(rolling_std.mean())
        feats["rolling_std_max"] = float(rolling_std.max())
        feats["rolling_q75_of_stds"] = float(np.quantile(rolling_std, 0.75))
    else:
        feats["rolling_std_mean"] = feats["std"]
        feats["rolling_std_max"] = feats["std"]
        feats["rolling_q75_of_stds"] = feats["std"]

    # --- dominio de frecuencia (FFT) ---
    fft_vals = rfft(signal - signal.mean())
    fft_mag = np.abs(fft_vals)
    freqs = rfftfreq(n, d=1.0 / SAMPLING_RATE_HZ)

    total_energy = float((fft_mag**2).sum()) + 1e-12
    dominant_idx = int(np.argmax(fft_mag[1:])) + 1  # excluye DC
    feats["dominant_frequency_hz"] = float(freqs[dominant_idx])
    feats["dominant_frequency_energy_ratio"] = float(fft_mag[dominant_idx] ** 2 / total_energy)
    feats["spectral_entropy"] = _spectral_entropy(fft_mag)
    feats["spectral_centroid_hz"] = float((freqs * fft_mag).sum() / (fft_mag.sum() + 1e-12))

    # energia por banda (bandas tipicas de defecto de rodamiento en 20kHz)
    bands = [(0, 500), (500, 2000), (2000, 5000), (5000, 10000)]
    for lo, hi in bands:
        mask = (freqs >= lo) & (freqs < hi)
        feats[f"energy_{lo}_{hi}hz"] = float((fft_mag[mask] ** 2).sum() / total_energy)

    return feats


def build_feature_table(snapshots: list[Snapshot]) -> pd.DataFrame:
    rows = []
    for snap in snapshots:
        row = extract_features(snap.signal)
        row["experiment"] = snap.experiment
        row["file_index"] = snap.file_index
        row["time_to_failure_min"] = time_to_failure_minutes(snap)
        row["rul_fraction"] = time_to_failure_fraction(snap)
        rows.append(row)
    return pd.DataFrame(rows)
