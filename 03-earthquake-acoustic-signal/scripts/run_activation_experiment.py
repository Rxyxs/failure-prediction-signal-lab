"""Ejecuta la comparacion de activaciones (ReLU/GELU/Swish) de la CNN 1D con
loss custom (Huber ponderada), y persiste metricas y figuras.

Genera una senal sintetica (misma semilla que ``scripts/generate_figures.py``
por defecto, pero configurable via CLI), la parte en segmentos, entrena la
CNN 1D con cada activacion usando ``WeightedHuberLoss``, y guarda:

- ``reports/activation_experiment.json``: MAE de validacion y loss por epoca
  para cada activacion.
- ``data/processed/features.duckdb`` (tabla ``activation_experiment``): mismo
  resultado en formato tabular, para consultarlo con SQL.
- ``reports/figures/activation_comparison.png``: barchart de MAE por activacion.
- ``reports/figures/activation_loss_curves.png``: loss de entrenamiento por epoca.

Ejecutar con: ``python -m scripts.run_activation_experiment``

Autor: Pablo Reyes
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from src.config import FEATURES_DB_PATH, REPORTS_DIR, SyntheticConfig, TrainingConfig, ensure_directories
from src.data.synthetic import generate_synthetic_dataset
from src.features.build_features import save_dataframe_to_duckdb
from src.features.windowing import create_windows
from src.models.activation_experiment import run_activation_comparison
from src.models.cnn_model import subsample_signal
from src.viz.plots import plot_activation_comparison, plot_activation_loss_curves

FIGURES_DIR = REPORTS_DIR / "figures"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Comparacion de activaciones para la CNN 1D.")
    parser.add_argument("--n-segments", type=int, default=24, help="Numero de segmentos sinteticos a generar.")
    parser.add_argument("--segment-size", type=int, default=150_000, help="Tamano de cada segmento.")
    parser.add_argument("--cnn-epochs", type=int, default=6, help="Numero de epocas de entrenamiento por activacion.")
    parser.add_argument("--val-fraction", type=float, default=0.25, help="Fraccion de segmentos usados para validacion.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    ensure_directories()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    synthetic_config = SyntheticConfig(n_segments=args.n_segments, segment_size=args.segment_size)
    raw_df = generate_synthetic_dataset(synthetic_config)
    windows = create_windows(raw_df, segment_size=args.segment_size)

    training_config = TrainingConfig(cnn_epochs=args.cnn_epochs)
    signals = np.stack(
        [subsample_signal(w["signal"], training_config.cnn_subsample) for w in windows]
    )
    targets = np.array([w["time_to_failure"] for w in windows], dtype=np.float32)

    n_val = max(1, int(len(signals) * args.val_fraction))
    train_signals, val_signals = signals[:-n_val], signals[-n_val:]
    train_targets, val_targets = targets[:-n_val], targets[-n_val:]

    print(f"Entrenando CNN 1D con {len(train_signals)} segmentos (train) y {len(val_signals)} (val) ...")
    results = run_activation_comparison(train_signals, train_targets, val_signals, val_targets, training_config)

    mae_by_activation = {name: r.val_mae for name, r in results.items()}
    epoch_losses_by_activation = {name: r.epoch_losses for name, r in results.items()}

    report = {
        "mae_by_activation": mae_by_activation,
        "epoch_losses_by_activation": epoch_losses_by_activation,
        "loss": "WeightedHuberLoss",
        "cnn_epochs": training_config.cnn_epochs,
        "n_train_segments": len(train_signals),
        "n_val_segments": len(val_signals),
    }
    report_path = REPORTS_DIR / "activation_experiment.json"
    with open(report_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    print(f"Reporte guardado en {report_path}")

    rows = []
    for name, result in results.items():
        for epoch, loss in enumerate(result.epoch_losses, start=1):
            rows.append({"activation": name, "epoch": epoch, "train_loss": loss, "val_mae": result.val_mae})
    save_dataframe_to_duckdb(pd.DataFrame(rows), FEATURES_DB_PATH, table_name="activation_experiment")
    print(f"Tabla 'activation_experiment' guardada en {FEATURES_DB_PATH}")

    plot_activation_comparison(mae_by_activation, FIGURES_DIR / "activation_comparison.png")
    plot_activation_loss_curves(epoch_losses_by_activation, FIGURES_DIR / "activation_loss_curves.png")
    print("Figuras guardadas en reports/figures/activation_comparison.png y activation_loss_curves.png")

    for name, mae in sorted(mae_by_activation.items(), key=lambda kv: kv[1]):
        print(f"  {name:8s} -> MAE val = {mae:.4f}")


if __name__ == "__main__":
    main()
