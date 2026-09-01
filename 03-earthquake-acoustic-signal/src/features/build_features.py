"""Orquesta la extraccion completa de features y su persistencia en DuckDB.

Combina las features de dominio temporal, frecuencia y Hilbert para cada
segmento generado por :mod:`src.features.windowing`, y escribe el resultado
en una tabla DuckDB en ``data/processed/features.duckdb``.

Autor: Pablo Reyes
"""
from __future__ import annotations

from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from src.config import FeatureConfig
from src.features.advanced_features import (
    extract_autocorrelation_features,
    extract_entropy_features,
    extract_peak_count_features,
)
from src.features.frequency_domain import (
    extract_fft_features,
    extract_hilbert_features,
    extract_spectrogram_features,
    extract_welch_psd_features,
)
from src.features.time_domain import extract_time_domain_features
from src.features.windowing import create_windows


def extract_segment_features(signal: np.ndarray, config: FeatureConfig) -> dict:
    """Extrae todas las features (tiempo, frecuencia, Hilbert, avanzadas) para un segmento."""
    features: dict = {}
    features.update(extract_time_domain_features(signal, n_rolling_windows=config.n_rolling_windows))
    features.update(extract_fft_features(signal))
    features.update(
        extract_welch_psd_features(signal, nperseg=config.welch_nperseg, n_bands=config.n_freq_bands)
    )
    features.update(extract_spectrogram_features(signal, nperseg=config.spectrogram_nperseg))
    features.update(extract_hilbert_features(signal))
    features.update(extract_entropy_features(signal))
    features.update(extract_autocorrelation_features(signal))
    features.update(extract_peak_count_features(signal))
    return features


def build_feature_table(raw_df: pd.DataFrame, config: FeatureConfig | None = None) -> pd.DataFrame:
    """Construye el DataFrame de features a partir de la senal cruda completa."""
    config = config or FeatureConfig()
    windows = create_windows(raw_df, segment_size=config.segment_size)

    rows = []
    for window in windows:
        row = extract_segment_features(window["signal"], config)
        row["segment_id"] = window["segment_id"]
        row["time_to_failure"] = window["time_to_failure"]
        rows.append(row)

    df = pd.DataFrame(rows)
    # Reordena para que segment_id y time_to_failure queden primero.
    ordered_cols = ["segment_id", "time_to_failure"] + [
        c for c in df.columns if c not in ("segment_id", "time_to_failure")
    ]
    return df[ordered_cols]


def save_features_to_duckdb(
    features_df: pd.DataFrame, db_path: Path, table_name: str = "segment_features"
) -> None:
    """Persiste el DataFrame de features en una tabla DuckDB, sobrescribiendo si existe."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(db_path))
    try:
        con.execute(f"DROP TABLE IF EXISTS {table_name}")
        con.execute(f"CREATE TABLE {table_name} AS SELECT * FROM features_df")
    finally:
        con.close()


def load_features_from_duckdb(db_path: Path, table_name: str = "segment_features") -> pd.DataFrame:
    """Lee la tabla de features desde DuckDB y la retorna como DataFrame."""
    con = duckdb.connect(str(db_path), read_only=True)
    try:
        return con.execute(f"SELECT * FROM {table_name}").fetchdf()
    finally:
        con.close()


def save_dataframe_to_duckdb(df: pd.DataFrame, db_path: Path, table_name: str) -> None:
    """Persiste un DataFrame generico en una tabla DuckDB, sobrescribiendo si existe.

    Complementa a ``save_features_to_duckdb`` para tablas auxiliares (p.ej.
    resultados de experimentos) que no siguen el esquema de segment_features.
    """
    db_path.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(db_path))
    try:
        con.execute(f"DROP TABLE IF EXISTS {table_name}")
        con.execute(f"CREATE TABLE {table_name} AS SELECT * FROM df")
    finally:
        con.close()
