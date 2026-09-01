"""Configuracion centralizada del proyecto lanl-earthquake-signal-prediction.

Define rutas y parametros por defecto usados a lo largo del pipeline:
generacion sintetica, extraccion de features, entrenamiento y API.

Autor: Pablo Reyes
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
REPORTS_DIR = ROOT_DIR / "reports"
MODELS_DIR = ROOT_DIR / "models_store"

RAW_SIGNAL_PARQUET = RAW_DATA_DIR / "synthetic_signal.parquet"
FEATURES_DB_PATH = PROCESSED_DATA_DIR / "features.duckdb"
FEATURES_TABLE = "segment_features"

SEGMENT_SIZE = 150_000  # muestras por segmento, tal como el dataset original de LANL


@dataclass
class SyntheticConfig:
    """Parametros del generador sintetico de senales acusticas."""

    n_segments: int = 40
    segment_size: int = SEGMENT_SIZE
    sampling_rate_hz: float = 4_000_000.0
    base_noise_std: float = 2.0
    quake_min_gap_segments: int = 4
    quake_max_gap_segments: int = 9
    random_seed: int = 42


@dataclass
class FeatureConfig:
    """Parametros de extraccion de features."""

    segment_size: int = SEGMENT_SIZE
    n_rolling_windows: int = 10
    welch_nperseg: int = 4096
    n_freq_bands: int = 5
    spectrogram_nperseg: int = 2048


@dataclass
class TrainingConfig:
    """Parametros de entrenamiento y validacion cruzada."""

    n_splits: int = 5
    random_seed: int = 42
    cnn_epochs: int = 8
    cnn_batch_size: int = 16
    cnn_learning_rate: float = 1e-3
    cnn_subsample: int = 6000  # puntos submuestreados de cada segmento para la CNN 1D
    enable_tuning: bool = True  # optimiza hiperparametros de LightGBM/CatBoost con Optuna
    n_tuning_trials: int = 15  # numero de trials de Optuna por modelo


def ensure_directories() -> None:
    """Crea los directorios de datos, reportes y modelos si no existen."""
    for directory in (RAW_DATA_DIR, PROCESSED_DATA_DIR, REPORTS_DIR, MODELS_DIR):
        directory.mkdir(parents=True, exist_ok=True)
