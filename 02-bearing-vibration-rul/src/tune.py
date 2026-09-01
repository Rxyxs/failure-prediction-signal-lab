"""Busqueda de hiperparametros con Optuna para CatBoost (mejor modelo del
baseline), optimizando MAE promedio en GroupKFold leave-one-experiment-out
-- el mismo protocolo de validacion que el pipeline principal, para que la
mejora sea comparable de forma honesta.

    python -m src.tune
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import optuna
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GroupKFold

from src.ingest import load_all_snapshots
from src.features import build_feature_table

ROOT = Path(__file__).resolve().parent.parent
REPORTS_DIR = ROOT / "outputs" / "reports"
MODELS_DIR = ROOT / "outputs" / "models"

N_TRIALS = 30
STRIDE = 3


def _objective(trial, X, y, groups):
    params = {
        "iterations": trial.suggest_int("iterations", 200, 800),
        "depth": trial.suggest_int("depth", 4, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 10.0),
        "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 1.0),
    }
    gkf = GroupKFold(n_splits=3)
    maes = []
    for train_idx, test_idx in gkf.split(X, y, groups):
        model = CatBoostRegressor(**params, random_seed=42, verbose=False, allow_writing_files=False)
        model.fit(X.iloc[train_idx], y[train_idx])
        pred = model.predict(X.iloc[test_idx])
        maes.append(mean_absolute_error(y[test_idx], pred))
    return float(np.mean(maes))


def main() -> None:
    print(f"[1/3] Cargando senal y features (stride={STRIDE})...")
    snapshots = load_all_snapshots(stride=STRIDE)
    table = build_feature_table(snapshots)
    feature_cols = [c for c in table.columns if c not in ("experiment", "file_index", "time_to_failure_min", "rul_fraction")]
    X = table[feature_cols]
    y = table["rul_fraction"].to_numpy()
    groups = table["experiment"].to_numpy()

    print(f"[2/3] Optuna: {N_TRIALS} trials, minimizando MAE promedio en GroupKFold (3 folds)...")
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(lambda t: _objective(t, X, y, groups), n_trials=N_TRIALS)

    print(f"\nMejor MAE (Optuna): {study.best_value:.4f}")
    print(f"Mejores parametros: {study.best_params}")

    print("[3/3] Reentrenando modelo final sobre todos los datos...")
    best_model = CatBoostRegressor(**study.best_params, random_seed=42, verbose=False, allow_writing_files=False)
    best_model.fit(X, y)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, MODELS_DIR / "catboost_tuned.joblib")
    joblib.dump(feature_cols, MODELS_DIR / "feature_order_tuned.joblib")

    baseline_metrics = json.load(open(REPORTS_DIR / "model_metrics.json", encoding="utf-8"))
    baseline_catboost_mae = next(m["mae_mean"] for m in baseline_metrics if m["model"] == "catboost" and m["target"] == "rul_fraction")

    result = {
        "baseline_catboost_mae": baseline_catboost_mae,
        "tuned_catboost_mae": round(study.best_value, 4),
        "improvement": round(baseline_catboost_mae - study.best_value, 4),
        "n_trials": N_TRIALS,
        "best_params": study.best_params,
    }
    with open(REPORTS_DIR / "optuna_tuning_result.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"\n=== Resultado ===")
    print(f"CatBoost baseline (sin tuning): MAE={baseline_catboost_mae}")
    print(f"CatBoost tuned (Optuna, {N_TRIALS} trials): MAE={study.best_value:.4f}")
    print(f"Mejora: {result['improvement']:+.4f}")
    print(f"\nGuardado en: {REPORTS_DIR / 'optuna_tuning_result.json'}")


if __name__ == "__main__":
    main()
