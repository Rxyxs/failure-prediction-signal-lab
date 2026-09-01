"""Pipeline aditivo para el 3er enfoque de modelado (MLP PyTorch, ReLU vs
GELU vs Swish): reusa la misma ingesta/features/target que src.pipeline
(no las recalcula desde cero salvo que sea necesario), agrega sus metricas
a DuckDB en una tabla separada y genera un grafico comparativo de
activaciones en outputs/reports/.

    python -m src.dl_pipeline
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from src.ingest import load_all_snapshots
from src.features import build_feature_table
from src.dl_model import evaluate_activations_group_kfold
from src.database import export_dl_metrics

ROOT = Path(__file__).resolve().parent.parent
REPORTS_DIR = ROOT / "outputs" / "reports"

STRIDE = 3  # misma reduccion que src.pipeline, mismo costo de computo


def plot_activation_comparison(results: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    colors = ["#8A5A2C", "#EB5E28", "#FFF000"]
    ax.bar(results["model"], results["mae_mean"], yerr=results["mae_std"], capsize=6, color=colors)
    ax.set_ylabel("MAE (fraccion de RUL)")
    ax.set_title("MLP (PyTorch): comparacion de activaciones\nGroupKFold leave-one-experiment-out")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    print(f"[1/3] Cargando senal cruda + features (stride={STRIDE}, mismo pipeline que src.pipeline)...")
    snapshots = load_all_snapshots(stride=STRIDE)
    table = build_feature_table(snapshots)

    feature_cols = [c for c in table.columns if c not in ("experiment", "file_index", "time_to_failure_min", "rul_fraction")]
    X = table[feature_cols]
    y = table["rul_fraction"].to_numpy()
    groups = table["experiment"].to_numpy()

    print("[2/3] MLP PyTorch: comparando activaciones ReLU / GELU / Swish (loss custom, GroupKFold)...")
    results = evaluate_activations_group_kfold(X, y, groups, n_splits=3)
    results.insert(0, "target", "rul_fraction")
    print("\n=== MAE (fraccion de vida) por activacion ===")
    print(results.to_string(index=False))

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    results.to_csv(REPORTS_DIR / "dl_activation_comparison.csv", index=False)
    with open(REPORTS_DIR / "dl_activation_comparison.json", "w", encoding="utf-8") as f:
        json.dump(results.to_dict(orient="records"), f, indent=2, ensure_ascii=False)

    print("[3/3] Grafico + DuckDB...")
    plot_activation_comparison(results, REPORTS_DIR / "dl_activation_comparison.png")
    export_dl_metrics(results)
    print(f"\nGuardado en: {REPORTS_DIR}, outputs/bearing.duckdb (tabla dl_metrics)")


if __name__ == "__main__":
    main()
