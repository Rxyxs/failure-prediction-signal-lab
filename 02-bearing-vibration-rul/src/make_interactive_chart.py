"""Genera un grafico Plotly interactivo (RUL fraccional predicha vs. real) a
partir de predicciones OUT-OF-FOLD reales: mismo GroupKFold leave-one-
experiment-out (3 experimentos NASA IMS reales) usado en pipeline.py, pero
guardando la prediccion por snapshot en vez de solo el MAE agregado.

    python -m src.make_interactive_chart

Autor: Pablo Reyes
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from sklearn.model_selection import GroupKFold

from src.ingest import load_all_snapshots
from src.features import build_feature_table
from src.modeling import build_models

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "outputs" / "interactive"
STRIDE = 3


def main() -> None:
    print(f"Cargando senal cruda (NASA IMS Bearing, stride={STRIDE})...")
    snapshots = load_all_snapshots(stride=STRIDE)
    table = build_feature_table(snapshots)

    feature_cols = [
        c for c in table.columns if c not in ("experiment", "file_index", "time_to_failure_min", "rul_fraction")
    ]
    X = table[feature_cols]
    y = table["rul_fraction"].to_numpy()
    groups = table["experiment"].to_numpy()

    gkf = GroupKFold(n_splits=3)
    oof_pred = np.zeros_like(y)
    for train_idx, test_idx in gkf.split(X, y, groups):
        model = build_models()["catboost"]
        model.fit(X.iloc[train_idx], y[train_idx])
        oof_pred[test_idx] = model.predict(X.iloc[test_idx])

    mae = float(np.abs(oof_pred - y).mean())
    print(f"CatBoost OOF MAE (rul_fraction, leave-one-experiment-out): {mae:.4f}")

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            line=dict(color="#999999", dash="dash", width=1.5),
            name="Perfect prediction",
            hoverinfo="skip",
        )
    )
    colors = {"1st_test": "#1f77b4", "2nd_test": "#ff7f0e", "4th_test": "#2ca02c"}
    for exp in ["1st_test", "2nd_test", "4th_test"]:
        mask = groups == exp
        fig.add_trace(
            go.Scatter(
                x=y[mask],
                y=oof_pred[mask],
                mode="markers",
                name=f"{exp} (held out, n={mask.sum()})",
                marker=dict(size=6, opacity=0.7, color=colors[exp], line=dict(width=0.4, color="rgba(0,0,0,0.3)")),
                hovertemplate=(
                    f"<b>{exp}</b><br>Actual RUL fraction: %{{x:.3f}}<br>"
                    "Predicted RUL fraction: %{y:.3f}<extra></extra>"
                ),
            )
        )

    fig.update_layout(
        title=(
            f"02 — Bearing vibration RUL fraction: predicted vs. actual "
            f"(CatBoost, GroupKFold leave-one-experiment-out, real NASA IMS data, "
            f"n={len(y)}, MAE={mae:.4f})"
        ),
        xaxis_title="Actual RUL fraction [0,1]",
        yaxis_title="Predicted RUL fraction [0,1]",
        template="plotly_white",
        width=900,
        height=650,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig.update_xaxes(range=[-0.02, 1.02])
    fig.update_yaxes(range=[-0.02, 1.02])

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / "rul-fraction-predicted-vs-actual.html"
    fig.write_html(out_path, include_plotlyjs="inline", full_html=True)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
