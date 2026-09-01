"""Genera todas las figuras de analisis del proyecto en ``reports/figures/``.

Requiere que se haya ejecutado antes ``python -m src.pipeline`` (o al menos la
generacion de datos sinteticos), de forma que existan los artefactos en
``models_store`` y los reportes en ``reports``. Produce:

- signal_vs_ttf.png: senal acustica cruda vs. time_to_failure.
- ttf_distribution.png: histograma de time_to_failure.
- spectrogram_example.png: espectrograma de un segmento de ejemplo.
- mae_comparison.png: barchart de MAE por modelo.
- prediction_vs_actual.png: scatter de prediccion vs. real (mejor modelo y ensamble).
- feature_importance_lightgbm.png / feature_importance_random_forest.png: top-20 features.
- cv_mae_by_fold.png: evolucion del MAE por fold de cross-validation.

Ejecutar con: ``python -m scripts.generate_figures``

Autor: Pablo Reyes
"""
from __future__ import annotations

import json

import joblib
import numpy as np

from src.config import FEATURES_DB_PATH, MODELS_DIR, REPORTS_DIR, SyntheticConfig, ensure_directories
from src.data.synthetic import generate_synthetic_dataset
from src.features.build_features import load_features_from_duckdb
from src.viz.plots import (
    plot_cv_mae_by_fold,
    plot_feature_importance,
    plot_mae_comparison,
    plot_prediction_vs_actual,
    plot_signal_vs_ttf,
    plot_spectrogram,
    plot_ttf_distribution,
)

FIGURES_DIR = REPORTS_DIR / "figures"


def main() -> None:
    ensure_directories()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    with open(REPORTS_DIR / "training_report.json", "r", encoding="utf-8") as fh:
        report = json.load(fh)

    oof_data = np.load(REPORTS_DIR / "oof_predictions.npz")
    feature_cols = joblib.load(MODELS_DIR / "feature_columns.joblib")

    # -- Regenera una senal sintetica de ejemplo (misma semilla que el pipeline
    # por defecto) solo para las figuras exploratorias, sin tocar los datos
    # usados en el entrenamiento persistido.
    raw_df = generate_synthetic_dataset(SyntheticConfig(n_segments=6, segment_size=150_000))
    signal = raw_df["acoustic_data"].to_numpy()
    ttf = raw_df["time_to_failure"].to_numpy()

    print("Generando signal_vs_ttf.png ...")
    plot_signal_vs_ttf(signal, ttf, FIGURES_DIR / "signal_vs_ttf.png", n_points=300_000)

    print("Generando ttf_distribution.png ...")
    features_df = load_features_from_duckdb(FEATURES_DB_PATH)
    plot_ttf_distribution(features_df["time_to_failure"].to_numpy(), FIGURES_DIR / "ttf_distribution.png")

    print("Generando spectrogram_example.png ...")
    plot_spectrogram(signal[:150_000], FIGURES_DIR / "spectrogram_example.png")

    print("Generando mae_comparison.png ...")
    plot_mae_comparison(report["mae_by_model"], FIGURES_DIR / "mae_comparison.png")

    print("Generando prediction_vs_actual.png ...")
    best_model = min(
        (name for name in report["mae_by_model"] if name != "ensemble"),
        key=lambda n: report["mae_by_model"][n],
    )
    y_true = oof_data["y_true"]
    preds_to_plot = {
        best_model: oof_data[f"oof_{best_model}"],
        "ensemble": oof_data["oof_ensemble"],
    }
    plot_prediction_vs_actual(y_true, preds_to_plot, FIGURES_DIR / "prediction_vs_actual.png")

    print("Generando feature_importance_*.png ...")
    for model_name, fname in (("lightgbm", "feature_importance_lightgbm.png"), ("random_forest", "feature_importance_random_forest.png")):
        model_path = MODELS_DIR / f"{model_name}.joblib"
        if not model_path.exists():
            continue
        model = joblib.load(model_path)
        if not hasattr(model, "feature_importances_"):
            continue
        plot_feature_importance(
            feature_cols,
            np.asarray(model.feature_importances_, dtype=np.float64),
            FIGURES_DIR / fname,
            top_n=20,
            title=f"Top 20 features - {model_name}",
        )

    print("Generando cv_mae_by_fold.png ...")
    fold_mae = report.get("fold_mae_by_model", {})
    if fold_mae:
        plot_cv_mae_by_fold(fold_mae, FIGURES_DIR / "cv_mae_by_fold.png")

    print(f"\nFiguras guardadas en {FIGURES_DIR}")


if __name__ == "__main__":
    main()
