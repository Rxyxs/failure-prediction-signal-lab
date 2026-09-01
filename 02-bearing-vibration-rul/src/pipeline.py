"""Pipeline end-to-end: ingesta (senal cruda) -> feature engineering
(FFT + estadisticos) -> iteracion de modelos -> GroupKFold -> DuckDB.

    python -m src.pipeline
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd

from src.ingest import load_all_snapshots
from src.features import build_feature_table
from src.modeling import evaluate_group_kfold, fit_final_model
from src.database import export_results

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "outputs" / "models"
REPORTS_DIR = ROOT / "outputs" / "reports"

STRIDE = 3  # 1 de cada 3 snapshots -- ~3150 snapshots, acota tiempo de computo


def main() -> None:
    print(f"[1/4] Cargando senal cruda (NASA IMS Bearing, stride={STRIDE})...")
    snapshots = load_all_snapshots(stride=STRIDE)
    print(f"  {len(snapshots)} snapshots cargados de 3 experimentos independientes")

    print("[2/4] Feature engineering (FFT, espectrogramas, cuartiles rodantes, kurtosis, entropia)...")
    table = build_feature_table(snapshots)

    feature_cols = [c for c in table.columns if c not in ("experiment", "file_index", "time_to_failure_min", "rul_fraction")]
    X = table[feature_cols]
    y_minutes = table["time_to_failure_min"].to_numpy()
    y_fraction = table["rul_fraction"].to_numpy()
    groups = table["experiment"].to_numpy()

    print("[3a/4] Primer intento: target = minutos absolutos hasta falla...")
    results_minutes = evaluate_group_kfold(X, y_minutes, groups, n_splits=3)
    results_minutes.insert(0, "target", "minutes_absolute")
    print("\n=== MAE (minutos) por modelo, target=minutos absolutos ===")
    print(results_minutes.to_string(index=False))
    print(
        "\n>>> Hallazgo real: MAE de miles de minutos. Los 3 experimentos duran radicalmente\n"
        ">>> distinto (1st~15d, 2nd~7d, 4th~44d) -- un modelo entrenado en 2 experimentos\n"
        ">>> nunca aprende la escala absoluta del tercero. Se corrige con RUL fraccional."
    )

    print("\n[3b/4] Correccion: target = RUL fraccional [0,1] (invariante a duracion absoluta)...")
    results = evaluate_group_kfold(X, y_fraction, groups, n_splits=3)
    results.insert(0, "target", "rul_fraction")
    print("\n=== MAE (fraccion de vida) por modelo, target=RUL fraccional ===")
    print(results.to_string(index=False))

    best_name = results.iloc[0]["model"]
    y = y_fraction
    print(f"\n[4/4] Mejor modelo: {best_name} -> ajustando sobre TODOS los datos para produccion...")
    final_model = fit_final_model(best_name, X, y)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(final_model, MODELS_DIR / "best_model.joblib")
    joblib.dump(feature_cols, MODELS_DIR / "feature_order.joblib")

    all_results = pd.concat([results_minutes, results], ignore_index=True)
    all_results.to_csv(REPORTS_DIR / "model_metrics.csv", index=False)
    with open(REPORTS_DIR / "model_metrics.json", "w", encoding="utf-8") as f:
        json.dump(all_results.to_dict(orient="records"), f, indent=2, ensure_ascii=False)

    export_results(table, all_results)
    print(f"\nGuardado en: {MODELS_DIR}, {REPORTS_DIR}, outputs/bearing.duckdb")


if __name__ == "__main__":
    main()
