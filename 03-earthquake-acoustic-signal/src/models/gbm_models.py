"""Modelos de gradient boosting: LightGBM y CatBoost, optimizados para MAE.

Autor: Pablo Reyes
"""
from __future__ import annotations

from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor


def build_lightgbm_model(random_seed: int = 42) -> LGBMRegressor:
    """Construye un LightGBM Regressor con objetivo MAE (regression_l1)."""
    return LGBMRegressor(
        objective="regression_l1",
        metric="mae",
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=31,
        max_depth=-1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=random_seed,
        verbosity=-1,
    )


def build_catboost_model(random_seed: int = 42) -> CatBoostRegressor:
    """Construye un CatBoost Regressor con funcion de perdida MAE."""
    return CatBoostRegressor(
        loss_function="MAE",
        eval_metric="MAE",
        iterations=300,
        learning_rate=0.05,
        depth=6,
        random_seed=random_seed,
        verbose=False,
    )
