"""Iteracion de modelos: Ridge/Lasso (baseline) -> Random Forest ->
LightGBM+CatBoost -> GroupKFold por experimento (evita fuga entre las 3
corridas fisicas independientes), metrica MAE (minutos)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso, Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GroupKFold

RANDOM_STATE = 42


def build_models() -> dict:
    return {
        "ridge": Ridge(alpha=1.0, random_state=RANDOM_STATE),
        "lasso": Lasso(alpha=0.1, random_state=RANDOM_STATE, max_iter=5000),
        "random_forest": RandomForestRegressor(
            n_estimators=300, max_depth=10, min_samples_leaf=5, random_state=RANDOM_STATE, n_jobs=-1
        ),
        "lightgbm": LGBMRegressor(
            n_estimators=500, num_leaves=31, learning_rate=0.03, subsample=0.8,
            colsample_bytree=0.8, random_state=RANDOM_STATE, verbose=-1,
        ),
        "catboost": CatBoostRegressor(
            iterations=500, depth=6, learning_rate=0.03, random_seed=RANDOM_STATE,
            verbose=False, allow_writing_files=False,
        ),
    }


def evaluate_group_kfold(X: pd.DataFrame, y: np.ndarray, groups: np.ndarray, n_splits: int = 3) -> pd.DataFrame:
    """GroupKFold con n_splits = n_grupos (leave-one-experiment-out): cada
    fold entrena en 2 experimentos y evalua en el tercero, nunca visto."""
    gkf = GroupKFold(n_splits=n_splits)
    models = build_models()
    results = []

    for name, model_template in models.items():
        fold_maes = []
        for fold, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups), start=1):
            model = build_models()[name]
            model.fit(X.iloc[train_idx], y[train_idx])
            pred = model.predict(X.iloc[test_idx])
            mae = mean_absolute_error(y[test_idx], pred)
            fold_maes.append(mae)
            print(f"  [{name}] fold {fold} (held-out group={groups[test_idx][0]}): MAE={mae:.4f}")
        results.append({"model": name, "mae_mean": float(np.mean(fold_maes)), "mae_std": float(np.std(fold_maes))})

    return pd.DataFrame(results).sort_values("mae_mean")


def fit_final_model(name: str, X: pd.DataFrame, y: np.ndarray):
    model = build_models()[name]
    model.fit(X, y)
    return model
