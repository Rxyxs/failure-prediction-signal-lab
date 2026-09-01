"""Orquesta el entrenamiento, la validacion cruzada y el ensamble final.

Entrena Ridge, Lasso, Random Forest, LightGBM, CatBoost y una CNN 1D usando
GroupKFold para obtener predicciones out-of-fold (OOF) sin fuga temporal,
calcula el MAE de cada modelo, arma un ensamble ponderado, reentrena cada
modelo sobre el dataset completo y persiste los artefactos en ``models_store``
junto con un reporte de metricas en ``reports``.

Autor: Pablo Reyes
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error

from src.config import MODELS_DIR, REPORTS_DIR, TrainingConfig
from src.features.build_features import load_features_from_duckdb
from src.features.windowing import create_windows
from src.models.baseline import build_lasso_model, build_random_forest_model, build_ridge_model
from src.models.cnn_model import subsample_signal, train_cnn_model, predict_cnn
from src.models.cross_validation import get_group_kfold_splits
from src.models.ensemble import compute_optimized_weights, weighted_ensemble_predict
from src.models.gbm_models import build_catboost_model, build_lightgbm_model
from src.models.tuning import (
    build_tuned_catboost,
    build_tuned_lightgbm,
    tune_catboost,
    tune_lightgbm,
)

NON_FEATURE_COLS = ("segment_id", "time_to_failure")


def _feature_matrix(features_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, list[str]]:
    feature_cols = [c for c in features_df.columns if c not in NON_FEATURE_COLS]
    X = features_df[feature_cols].to_numpy(dtype=np.float64)
    y = features_df["time_to_failure"].to_numpy(dtype=np.float64)
    return X, y, feature_cols


def run_tabular_cv(
    model_builder, X: np.ndarray, y: np.ndarray, splits, random_seed: int
) -> tuple[np.ndarray, float, list[float]]:
    """Entrena ``model_builder`` en cada fold y arma el vector de predicciones OOF.

    Ademas del MAE global OOF, retorna el MAE individual de cada fold para
    poder graficar la evolucion del error a lo largo de la cross-validation.
    """
    oof_preds = np.zeros_like(y)
    fold_maes: list[float] = []
    for train_idx, val_idx in splits:
        model = model_builder(random_seed=random_seed)
        model.fit(X[train_idx], y[train_idx])
        fold_pred = model.predict(X[val_idx])
        oof_preds[val_idx] = fold_pred
        fold_maes.append(float(mean_absolute_error(y[val_idx], fold_pred)))
    mae = mean_absolute_error(y, oof_preds)
    return oof_preds, mae, fold_maes


def run_cnn_cv(
    raw_signals: np.ndarray, y: np.ndarray, splits, config: TrainingConfig
) -> tuple[np.ndarray, float]:
    """Entrena la CNN 1D en cada fold sobre la senal cruda submuestreada."""
    oof_preds = np.zeros_like(y)
    for train_idx, val_idx in splits:
        _model, _ = train_cnn_model(
            raw_signals[train_idx], y[train_idx], raw_signals[val_idx], y[val_idx], config=config
        )
        oof_preds[val_idx] = predict_cnn(_model, raw_signals[val_idx])
    mae = mean_absolute_error(y, oof_preds)
    return oof_preds, mae


def run_full_training(
    db_path: Path,
    training_config: TrainingConfig | None = None,
    raw_df: pd.DataFrame | None = None,
    segment_size: int | None = None,
) -> dict:
    """Ejecuta el flujo completo de entrenamiento, validacion y ensamble.

    Si se provee ``raw_df`` y ``segment_size``, tambien se entrena la CNN 1D
    sobre la forma de onda cruda; de lo contrario se omite ese modelo.
    """
    training_config = training_config or TrainingConfig()
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    features_df = load_features_from_duckdb(db_path)
    X, y, feature_cols = _feature_matrix(features_df)

    splits, _groups = get_group_kfold_splits(len(y), n_splits=training_config.n_splits)

    mae_by_model: dict[str, float] = {}
    oof_by_model: dict[str, np.ndarray] = {}
    fold_mae_by_model: dict[str, list[float]] = {}

    tabular_builders = {
        "ridge": build_ridge_model,
        "lasso": build_lasso_model,
        "random_forest": build_random_forest_model,
        "lightgbm": build_lightgbm_model,
        "catboost": build_catboost_model,
    }

    tuned_params: dict[str, dict] = {}
    if training_config.enable_tuning:
        lgbm_best = tune_lightgbm(
            X, y, splits, n_trials=training_config.n_tuning_trials, random_seed=training_config.random_seed
        )
        catboost_best = tune_catboost(
            X, y, splits, n_trials=training_config.n_tuning_trials, random_seed=training_config.random_seed
        )
        tuned_params = {"lightgbm": lgbm_best, "catboost": catboost_best}
        tabular_builders["lightgbm"] = lambda random_seed: build_tuned_lightgbm(
            lgbm_best, random_seed=random_seed
        )
        tabular_builders["catboost"] = lambda random_seed: build_tuned_catboost(
            catboost_best, random_seed=random_seed
        )

    for name, builder in tabular_builders.items():
        oof_preds, mae, fold_maes = run_tabular_cv(builder, X, y, splits, training_config.random_seed)
        mae_by_model[name] = mae
        oof_by_model[name] = oof_preds
        fold_mae_by_model[name] = fold_maes

    cnn_included = raw_df is not None and segment_size is not None
    if cnn_included:
        windows = create_windows(raw_df, segment_size=segment_size)
        raw_signals = np.stack(
            [subsample_signal(w["signal"], training_config.cnn_subsample) for w in windows]
        )
        raw_signals = (raw_signals - raw_signals.mean()) / (raw_signals.std() + 1e-8)
        oof_preds, mae = run_cnn_cv(raw_signals, y, splits, training_config)
        mae_by_model["cnn_1d"] = mae
        oof_by_model["cnn_1d"] = oof_preds

    # Ensamble con pesos optimizados por minimos cuadrados no negativos (NNLS)
    # sobre las predicciones out-of-fold, en vez de una heuristica fija basada
    # solo en el MAE individual de cada modelo.
    weights = compute_optimized_weights(oof_by_model, y)
    ensemble_oof = weighted_ensemble_predict(oof_by_model, weights)
    mae_by_model["ensemble"] = mean_absolute_error(y, ensemble_oof)

    # Reentrenamiento final sobre el dataset completo y persistencia de artefactos.
    trained_models = {}
    for name, builder in tabular_builders.items():
        model = builder(random_seed=training_config.random_seed)
        model.fit(X, y)
        trained_models[name] = model
        joblib.dump(model, MODELS_DIR / f"{name}.joblib")

    if cnn_included:
        final_cnn, _ = train_cnn_model(raw_signals, y, raw_signals, y, config=training_config)
        torch.save(final_cnn.state_dict(), MODELS_DIR / "cnn_1d.pt")

    joblib.dump(feature_cols, MODELS_DIR / "feature_columns.joblib")
    joblib.dump(weights, MODELS_DIR / "ensemble_weights.joblib")

    # Persistencia de predicciones OOF y verdad terreno para poder regenerar
    # graficos de analisis (scatter, MAE por fold) sin re-entrenar.
    oof_to_save = {f"oof_{name}": preds for name, preds in oof_by_model.items()}
    oof_to_save["oof_ensemble"] = ensemble_oof
    oof_to_save["y_true"] = y
    np.savez(REPORTS_DIR / "oof_predictions.npz", **oof_to_save)

    report = {
        "mae_by_model": mae_by_model,
        "ensemble_weights": weights,
        "tuned_hyperparameters": tuned_params,
        "fold_mae_by_model": fold_mae_by_model,
        "n_segments": int(len(y)),
        "n_features": int(len(feature_cols)),
        "n_splits": training_config.n_splits,
    }

    with open(REPORTS_DIR / "training_report.json", "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)

    mae_df = pd.DataFrame(
        {"model": list(mae_by_model.keys()), "mae": list(mae_by_model.values())}
    ).sort_values("mae")
    mae_df.to_csv(REPORTS_DIR / "mae_report.csv", index=False)

    return report
