"""Orquesta el flujo end-to-end: datos -> features -> entrenamiento -> reportes.

Ejecutable directamente con ``python -m src.pipeline``. Por defecto usa
parametros reducidos de la generacion sintetica para que el pipeline complete
en tiempo razonable; permite escalar el volumen de datos via flags de linea
de comandos.

Autor: Pablo Reyes
"""
from __future__ import annotations

import argparse
import time

from src.config import (
    FEATURES_DB_PATH,
    FeatureConfig,
    RAW_DATA_DIR,
    SyntheticConfig,
    TrainingConfig,
    ensure_directories,
)
from src.data.synthetic import generate_synthetic_dataset, maybe_download_real_dataset
from src.features.build_features import build_feature_table, save_features_to_duckdb
from src.models.train import run_full_training


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parsea los argumentos de linea de comandos del pipeline."""
    parser = argparse.ArgumentParser(description="Pipeline end-to-end de LANL earthquake signal prediction.")
    parser.add_argument("--n-segments", type=int, default=40, help="Numero de segmentos sinteticos a generar.")
    parser.add_argument(
        "--segment-size", type=int, default=150_000, help="Tamano de cada segmento (muestras)."
    )
    parser.add_argument("--n-splits", type=int, default=5, help="Numero de folds para GroupKFold.")
    parser.add_argument(
        "--cnn-epochs", type=int, default=8, help="Numero de epocas de entrenamiento de la CNN 1D."
    )
    parser.add_argument(
        "--skip-cnn", action="store_true", help="Omite el entrenamiento de la CNN 1D para acelerar el pipeline."
    )
    parser.add_argument(
        "--try-kaggle", action="store_true", help="Intenta descargar el dataset real de Kaggle si hay credenciales."
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> dict:
    """Ejecuta el pipeline completo y retorna el reporte de metricas final."""
    args = parse_args(argv)
    ensure_directories()

    t0 = time.time()
    print("== Paso 1/3: generacion / ingesta de datos ==")

    real_dataset_downloaded = False
    if args.try_kaggle:
        real_dataset_downloaded = maybe_download_real_dataset(RAW_DATA_DIR)

    if real_dataset_downloaded:
        print("Dataset real de Kaggle descargado correctamente en", RAW_DATA_DIR)
        raise SystemExit(
            "La ingesta del dataset real de Kaggle (CSV multi-GB) requiere un loader dedicado; "
            "use el modo sintetico por defecto para el pipeline de demostracion."
        )

    synthetic_config = SyntheticConfig(n_segments=args.n_segments, segment_size=args.segment_size)
    raw_df = generate_synthetic_dataset(synthetic_config)
    print(f"Datos sinteticos generados: {len(raw_df):,} muestras en {time.time() - t0:.1f}s")

    print("== Paso 2/3: extraccion de features ==")
    t1 = time.time()
    feature_config = FeatureConfig(segment_size=args.segment_size)
    features_df = build_feature_table(raw_df, feature_config)
    save_features_to_duckdb(features_df, FEATURES_DB_PATH)
    print(
        f"Features extraidas para {len(features_df)} segmentos "
        f"({features_df.shape[1] - 2} columnas) en {time.time() - t1:.1f}s"
    )

    print("== Paso 3/3: entrenamiento, validacion cruzada y ensamble ==")
    t2 = time.time()
    training_config = TrainingConfig(n_splits=args.n_splits, cnn_epochs=args.cnn_epochs)
    report = run_full_training(
        db_path=FEATURES_DB_PATH,
        training_config=training_config,
        raw_df=None if args.skip_cnn else raw_df,
        segment_size=None if args.skip_cnn else args.segment_size,
    )
    print(f"Entrenamiento completado en {time.time() - t2:.1f}s")

    print("\n=== Reporte de MAE por modelo ===")
    for name, mae in sorted(report["mae_by_model"].items(), key=lambda kv: kv[1]):
        print(f"  {name:15s} MAE = {mae:.4f}")

    print(f"\nPipeline completo en {time.time() - t0:.1f}s")
    return report


if __name__ == "__main__":
    main()
