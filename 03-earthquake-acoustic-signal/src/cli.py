"""CLI del proyecto lanl-earthquake-signal-prediction basado en Typer.

Provee subcomandos independientes para generar datos, extraer features,
entrenar/reentrenar modelos y hacer scoring en lote sobre un archivo CSV de
senales acusticas.

Autor: Pablo Reyes
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import typer

from src.config import (
    FEATURES_DB_PATH,
    FeatureConfig,
    MODELS_DIR,
    RAW_SIGNAL_PARQUET,
    SyntheticConfig,
    TrainingConfig,
    ensure_directories,
)
from src.data.synthetic import generate_synthetic_dataset
from src.features.build_features import build_feature_table, extract_segment_features, save_features_to_duckdb
from src.models.train import run_full_training

app = typer.Typer(help="CLI de lanl-earthquake-signal-prediction.")


@app.command()
def generate_data(
    n_segments: int = typer.Option(40, help="Numero de segmentos sinteticos a generar."),
    segment_size: int = typer.Option(150_000, help="Tamano de cada segmento (muestras)."),
) -> None:
    """Genera el dataset sintetico y lo guarda como parquet en data/raw/."""
    ensure_directories()
    config = SyntheticConfig(n_segments=n_segments, segment_size=segment_size)
    df = generate_synthetic_dataset(config)
    df.to_parquet(RAW_SIGNAL_PARQUET)
    typer.echo(f"Dataset sintetico generado con {len(df):,} muestras -> {RAW_SIGNAL_PARQUET}")


@app.command()
def extract_features(
    segment_size: int = typer.Option(150_000, help="Tamano de cada segmento (muestras)."),
) -> None:
    """Extrae features a partir del dataset crudo y las guarda en DuckDB."""
    ensure_directories()
    if not RAW_SIGNAL_PARQUET.exists():
        typer.echo("No existe dataset crudo; genere datos primero con 'generate-data'.")
        raise typer.Exit(code=1)

    raw_df = pd.read_parquet(RAW_SIGNAL_PARQUET)
    feature_config = FeatureConfig(segment_size=segment_size)
    features_df = build_feature_table(raw_df, feature_config)
    save_features_to_duckdb(features_df, FEATURES_DB_PATH)
    typer.echo(f"Features guardadas en {FEATURES_DB_PATH} ({len(features_df)} segmentos).")


@app.command()
def train(
    n_splits: int = typer.Option(5, help="Numero de folds para GroupKFold."),
    skip_cnn: bool = typer.Option(False, help="Omite el entrenamiento de la CNN 1D."),
) -> None:
    """Entrena/reentrena todos los modelos y guarda reportes de MAE."""
    ensure_directories()
    raw_df = pd.read_parquet(RAW_SIGNAL_PARQUET) if (RAW_SIGNAL_PARQUET.exists() and not skip_cnn) else None
    training_config = TrainingConfig(n_splits=n_splits)
    report = run_full_training(
        db_path=FEATURES_DB_PATH,
        training_config=training_config,
        raw_df=raw_df,
        segment_size=150_000 if raw_df is not None else None,
    )
    typer.echo(json.dumps(report["mae_by_model"], indent=2))


@app.command()
def score_batch(
    input_csv: Path = typer.Argument(..., help="CSV con una columna 'acoustic_data' por segmento a scorear."),
    output_csv: Path = typer.Option(Path("reports/batch_scores.csv"), help="Ruta de salida de las predicciones."),
) -> None:
    """Realiza scoring en lote sobre uno o mas segmentos contenidos en un CSV."""
    if not input_csv.exists():
        typer.echo(f"No se encontro el archivo {input_csv}")
        raise typer.Exit(code=1)

    feature_cols_path = MODELS_DIR / "feature_columns.joblib"
    model_path = MODELS_DIR / "lightgbm.joblib"
    if not model_path.exists():
        model_path = MODELS_DIR / "ridge.joblib"
    if not (model_path.exists() and feature_cols_path.exists()):
        typer.echo("No hay modelos entrenados disponibles. Ejecute 'train' primero.")
        raise typer.Exit(code=1)

    model = joblib.load(model_path)
    feature_cols = joblib.load(feature_cols_path)

    df = pd.read_csv(input_csv)
    signal = df["acoustic_data"].to_numpy(dtype=np.float64)
    features = extract_segment_features(signal, FeatureConfig())
    feature_vector = np.array([[features.get(col, 0.0) for col in feature_cols]])
    prediction = float(model.predict(feature_vector)[0])

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{"input_file": str(input_csv), "predicted_time_to_failure": prediction}]).to_csv(
        output_csv, index=False
    )
    typer.echo(f"Prediccion: {prediction:.4f} -> guardada en {output_csv}")


if __name__ == "__main__":
    app()
