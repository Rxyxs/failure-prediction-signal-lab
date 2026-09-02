"""Genera un grafico Plotly interactivo (predicho vs. real, time_to_failure)
a partir de las predicciones OOF REALES guardadas por run_full_training en
reports/oof_predictions.npz (mismo articulo de datos que reports/figures/
prediction_vs_actual.png, en version interactiva y autocontenida).

    python -m scripts.generate_interactive_chart

Autor: Pablo Reyes
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import plotly.graph_objects as go

REPORTS_DIR = Path(__file__).resolve().parent.parent / "reports"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "outputs" / "interactive"

MODEL_LABELS = {
    "oof_ridge": "Ridge",
    "oof_lasso": "Lasso",
    "oof_random_forest": "Random Forest",
    "oof_lightgbm": "LightGBM (Optuna)",
    "oof_catboost": "CatBoost (Optuna)",
    "oof_cnn_1d": "1D CNN",
    "oof_ensemble": "NNLS ensemble",
}


def main() -> None:
    data = np.load(REPORTS_DIR / "oof_predictions.npz")
    y_true = data["y_true"]

    fig = go.Figure()
    max_val = float(y_true.max())
    min_val = float(min(y_true.min(), 0))
    for key in ["oof_ridge", "oof_lasso", "oof_random_forest", "oof_cnn_1d", "oof_lightgbm", "oof_catboost", "oof_ensemble"]:
        preds = data[key]
        mae = float(np.abs(preds - y_true).mean())
        label = MODEL_LABELS[key]
        visible = True if key in ("oof_ensemble", "oof_catboost", "oof_lightgbm") else "legendonly"
        fig.add_trace(
            go.Scatter(
                x=y_true,
                y=preds,
                mode="markers",
                name=f"{label} (MAE={mae:.3f})",
                marker=dict(size=9, opacity=0.75, line=dict(width=0.5, color="rgba(0,0,0,0.35)")),
                visible=visible,
                hovertemplate=(
                    f"<b>{label}</b><br>Actual TTF: %{{x:.3f}}<br>"
                    "Predicted TTF: %{y:.3f}<extra></extra>"
                ),
            )
        )
        max_val = max(max_val, float(preds.max()))
        min_val = min(min_val, float(preds.min()))

    pad = (max_val - min_val) * 0.05
    fig.add_trace(
        go.Scatter(
            x=[min_val - pad, max_val + pad],
            y=[min_val - pad, max_val + pad],
            mode="lines",
            line=dict(color="#999999", dash="dash", width=1.5),
            name="Perfect prediction",
            hoverinfo="skip",
        )
    )

    fig.update_layout(
        title=(
            f"03 — LANL-style acoustic signal: predicted vs. actual time_to_failure "
            f"(out-of-fold, GroupKFold n_splits=4, n={len(y_true)} segments)"
        ),
        xaxis_title="Actual time_to_failure",
        yaxis_title="Predicted time_to_failure",
        template="plotly_white",
        width=900,
        height=650,
        legend=dict(title="Model (click to toggle)"),
    )
    fig.update_xaxes(range=[min_val - pad, max_val + pad])
    fig.update_yaxes(range=[min_val - pad, max_val + pad])

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / "ttf-predicted-vs-actual.html"
    fig.write_html(out_path, include_plotlyjs="inline", full_html=True)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
