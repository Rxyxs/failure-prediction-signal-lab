"""Features de dominio de frecuencia y transformada de Hilbert.

Incluye FFT (energia y frecuencia dominante), densidad espectral de potencia
via Welch, energia por bandas de frecuencia, estadisticos del espectrograma,
y la envolvente/fase instantanea obtenidas via la transformada de Hilbert.

Autor: Pablo Reyes
"""
from __future__ import annotations

import numpy as np
from scipy.signal import hilbert, spectrogram, welch


def extract_fft_features(signal: np.ndarray) -> dict:
    """Calcula features basadas en la magnitud del espectro de Fourier."""
    signal = signal.astype(np.float64)
    n = len(signal)

    fft_vals = np.fft.rfft(signal)
    fft_mag = np.abs(fft_vals)
    freqs = np.fft.rfftfreq(n)

    total_energy = float(np.sum(fft_mag ** 2))
    dominant_idx = int(np.argmax(fft_mag[1:]) + 1) if len(fft_mag) > 1 else 0

    return {
        "fft_energy": total_energy,
        "fft_mean_magnitude": float(np.mean(fft_mag)),
        "fft_std_magnitude": float(np.std(fft_mag)),
        "fft_max_magnitude": float(np.max(fft_mag)),
        "fft_dominant_freq": float(freqs[dominant_idx]),
        "fft_spectral_centroid": float(
            np.sum(freqs * fft_mag) / (np.sum(fft_mag) + 1e-12)
        ),
    }


def extract_welch_psd_features(signal: np.ndarray, nperseg: int = 4096, n_bands: int = 5) -> dict:
    """Calcula la densidad espectral de potencia (Welch) y energia por bandas."""
    signal = signal.astype(np.float64)
    nperseg = min(nperseg, len(signal))

    freqs, psd = welch(signal, nperseg=nperseg)

    features = {
        "psd_mean": float(np.mean(psd)),
        "psd_std": float(np.std(psd)),
        "psd_max": float(np.max(psd)),
        "psd_total_power": float(np.sum(psd)),
    }

    # Energia por bandas de frecuencia equiespaciadas entre 0 y Nyquist.
    max_freq = freqs[-1] if len(freqs) > 0 else 1.0
    band_edges = np.linspace(0, max_freq, n_bands + 1)
    for i in range(n_bands):
        mask = (freqs >= band_edges[i]) & (freqs < band_edges[i + 1])
        band_energy = float(np.sum(psd[mask])) if mask.any() else 0.0
        features[f"psd_band_{i}_energy"] = band_energy

    return features


def extract_spectrogram_features(signal: np.ndarray, nperseg: int = 2048) -> dict:
    """Calcula estadisticos resumidos del espectrograma de la senal."""
    signal = signal.astype(np.float64)
    nperseg = min(nperseg, len(signal))
    if nperseg < 8:
        return {"spec_mean": 0.0, "spec_std": 0.0, "spec_max": 0.0}

    _, _, sxx = spectrogram(signal, nperseg=nperseg)
    return {
        "spec_mean": float(np.mean(sxx)),
        "spec_std": float(np.std(sxx)),
        "spec_max": float(np.max(sxx)),
    }


def extract_hilbert_features(signal: np.ndarray) -> dict:
    """Calcula features de envolvente y fase instantanea via Hilbert."""
    signal = signal.astype(np.float64)
    analytic_signal = hilbert(signal)
    envelope = np.abs(analytic_signal)
    instantaneous_phase = np.unwrap(np.angle(analytic_signal))
    instantaneous_freq = np.diff(instantaneous_phase) / (2.0 * np.pi)

    return {
        "hilbert_envelope_mean": float(np.mean(envelope)),
        "hilbert_envelope_std": float(np.std(envelope)),
        "hilbert_envelope_max": float(np.max(envelope)),
        "hilbert_phase_std": float(np.std(instantaneous_phase)),
        "hilbert_inst_freq_mean": float(np.mean(instantaneous_freq)) if instantaneous_freq.size else 0.0,
        "hilbert_inst_freq_std": float(np.std(instantaneous_freq)) if instantaneous_freq.size else 0.0,
    }
