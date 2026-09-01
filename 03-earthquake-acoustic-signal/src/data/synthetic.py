"""Generador sintetico de alta fidelidad de senales acusticas tipo LANL.

Simula el comportamiento observado en el dataset "LANL Earthquake Prediction"
de Kaggle: una senal acustica de alta frecuencia con ruido de fondo, picos de
energia (precursores) que crecen en amplitud a medida que se acerca una falla,
y una caida abrupta de la variable objetivo ``time_to_failure`` justo despues
de cada evento sismico simulado. El objetivo es permitir que todo el pipeline
(`python -m src.pipeline`) corra de forma autonoma sin depender del dataset
real, que pesa varios gigabytes.

Autor: Pablo Reyes
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import SyntheticConfig


def _generate_single_cycle(
    rng: np.random.Generator,
    n_segments: int,
    segment_size: int,
    base_noise_std: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Genera un ciclo completo entre dos fallas consecutivas.

    Devuelve la senal acustica cruda y el vector de ``time_to_failure``
    correspondiente, ambos de longitud ``n_segments * segment_size``.
    """
    total_points = n_segments * segment_size
    t = np.arange(total_points)

    # time_to_failure decae linealmente desde un maximo aleatorio hasta 0.
    cycle_duration = rng.uniform(8.0, 16.0)
    time_to_failure = cycle_duration * (1.0 - t / total_points)

    # Ruido base gaussiano de fondo (componente estocastica de baja amplitud).
    signal = rng.normal(loc=0.0, scale=base_noise_std, size=total_points)

    # Deriva estocastica (random walk suavizado) que modula la amplitud base.
    drift = np.cumsum(rng.normal(0.0, 0.05, size=total_points))
    drift = drift - np.linspace(drift[0], drift[-1], total_points)  # remueve tendencia lineal
    signal += drift * 0.5

    # Precursores: a medida que time_to_failure disminuye, aumenta la
    # probabilidad e intensidad de micro-eventos sismicos (picos de energia).
    proximity = 1.0 - (time_to_failure / cycle_duration)  # 0 -> lejos, 1 -> cerca de la falla
    n_precursors = int(30 + 120 * proximity.mean())
    precursor_positions = rng.choice(total_points, size=n_precursors, replace=False)
    for pos in precursor_positions:
        local_proximity = proximity[pos]
        amplitude = base_noise_std * (3.0 + 25.0 * local_proximity) * rng.uniform(0.5, 1.5)
        width = rng.integers(5, 60)
        start = max(0, pos - width)
        end = min(total_points, pos + width)
        window = np.arange(start, end) - pos
        pulse = amplitude * np.exp(-0.5 * (window / (width / 3.0)) ** 2) * rng.choice([-1.0, 1.0])
        signal[start:end] += pulse

    # Evento principal de falla: pulso de gran amplitud justo antes de t=0.
    failure_idx = total_points - 1
    quake_width = rng.integers(200, 800)
    start = max(0, failure_idx - quake_width)
    quake_amplitude = base_noise_std * rng.uniform(40.0, 90.0)
    quake_window = np.arange(start, total_points) - failure_idx
    quake_pulse = quake_amplitude * np.exp(-0.5 * (quake_window / (quake_width / 4.0)) ** 2)
    quake_pulse *= np.sign(rng.normal(size=quake_pulse.shape))
    signal[start:total_points] += quake_pulse

    return signal.astype(np.float32), time_to_failure.astype(np.float64)


def generate_synthetic_dataset(config: SyntheticConfig | None = None) -> pd.DataFrame:
    """Genera un DataFrame sintetico con columnas ``acoustic_data`` y ``time_to_failure``.

    La senal se compone de multiples ciclos consecutivos entre fallas, cada uno
    generado con :func:`_generate_single_cycle`, concatenados para simular un
    registro continuo como el del dataset original de LANL.
    """
    config = config or SyntheticConfig()
    rng = np.random.default_rng(config.random_seed)

    signals: list[np.ndarray] = []
    ttf: list[np.ndarray] = []

    segments_generated = 0
    while segments_generated < config.n_segments:
        cycle_segments = int(
            rng.integers(config.quake_min_gap_segments, config.quake_max_gap_segments + 1)
        )
        cycle_segments = min(cycle_segments, config.n_segments - segments_generated)
        if cycle_segments <= 0:
            break
        sig, t2f = _generate_single_cycle(
            rng=rng,
            n_segments=cycle_segments,
            segment_size=config.segment_size,
            base_noise_std=config.base_noise_std,
        )
        signals.append(sig)
        ttf.append(t2f)
        segments_generated += cycle_segments

    acoustic_data = np.concatenate(signals)
    time_to_failure = np.concatenate(ttf)

    return pd.DataFrame(
        {
            "acoustic_data": acoustic_data,
            "time_to_failure": time_to_failure,
        }
    )


def maybe_download_real_dataset(destination_dir) -> bool:
    """Intenta descargar el dataset real de Kaggle si hay credenciales disponibles.

    Busca ``~/.kaggle/kaggle.json`` o las variables de entorno
    ``KAGGLE_USERNAME``/``KAGGLE_KEY``. Si no encuentra credenciales o falla la
    descarga por cualquier motivo, retorna ``False`` sin lanzar excepcion, de
    forma que el pipeline pueda continuar con el generador sintetico.
    """
    import os
    from pathlib import Path

    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    has_env_creds = bool(os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"))

    if not kaggle_json.exists() and not has_env_creds:
        return False

    try:
        import kaggle  # type: ignore

        destination_dir = Path(destination_dir)
        destination_dir.mkdir(parents=True, exist_ok=True)
        kaggle.api.authenticate()
        kaggle.api.competition_download_files(
            "LANL-Earthquake-Prediction", path=str(destination_dir), quiet=True
        )
        return True
    except Exception:
        return False
