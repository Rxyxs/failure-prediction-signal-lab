"""Pruebas de integridad de escritura/lectura de la tabla de features en DuckDB.

Autor: Pablo Reyes
"""
from __future__ import annotations

import pandas as pd
import pytest

from src.config import FeatureConfig
from src.data.synthetic import generate_synthetic_dataset
from src.features.build_features import build_feature_table, load_features_from_duckdb, save_features_to_duckdb
from src.config import SyntheticConfig


@pytest.fixture()
def sample_features_df() -> pd.DataFrame:
    synthetic_config = SyntheticConfig(n_segments=3, segment_size=2000, random_seed=1)
    raw_df = generate_synthetic_dataset(synthetic_config)
    feature_config = FeatureConfig(segment_size=2000, welch_nperseg=256, spectrogram_nperseg=128)
    return build_feature_table(raw_df, feature_config)


def test_save_and_load_roundtrip(tmp_path, sample_features_df):
    db_path = tmp_path / "features_test.duckdb"
    save_features_to_duckdb(sample_features_df, db_path)

    assert db_path.exists()

    loaded_df = load_features_from_duckdb(db_path)
    assert len(loaded_df) == len(sample_features_df)
    assert set(sample_features_df.columns) == set(loaded_df.columns)
    pd.testing.assert_frame_equal(
        sample_features_df.sort_values("segment_id").reset_index(drop=True),
        loaded_df.sort_values("segment_id").reset_index(drop=True),
        check_like=True,
    )


def test_save_overwrites_existing_table(tmp_path, sample_features_df):
    db_path = tmp_path / "features_test.duckdb"
    save_features_to_duckdb(sample_features_df, db_path)
    save_features_to_duckdb(sample_features_df, db_path)  # segunda escritura no debe fallar

    loaded_df = load_features_from_duckdb(db_path)
    assert len(loaded_df) == len(sample_features_df)


def test_time_to_failure_column_present(tmp_path, sample_features_df):
    db_path = tmp_path / "features_test.duckdb"
    save_features_to_duckdb(sample_features_df, db_path)
    loaded_df = load_features_from_duckdb(db_path)
    assert "time_to_failure" in loaded_df.columns
    assert loaded_df["time_to_failure"].notna().all()
