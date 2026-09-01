"""Optimizacion de hiperparametros para LightGBM y CatBoost via Optuna.

Se optimiza directamente el MAE de validacion cruzada (GroupKFold), usando el
mismo esquema de particionado temporal que el resto del pipeline para evitar
fuga de informacion entre train y validation. El numero de trials es
deliberadamente moderado para que la optimizacion complete en segundos sobre
el dataset sintetico de demostracion, pero la funcion es igualmente valida
sobre un dataset de mayor escala (solo requiere subir ``n_trials``).

Autor: Pablo Reyes
"""
from __future__ import annotations

import numpy as np
import optuna
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error

optuna.logging.set_verbosity(optuna.logging.WARNING)


def _cv_mae(model_builder, X: np.ndarray, y: np.ndarray, splits) -> float:
    """Calcula el MAE promedio out-of-fold para un ``model_builder`` dado."""
    oof = np.zeros_like(y)
    for train_idx, val_idx in splits:
        model = model_builder()
        model.fit(X[train_idx], y[train_idx])
        oof[val_idx] = model.predict(X[val_idx])
    return float(mean_absolute_error(y, oof))


def tune_lightgbm(
    X: np.ndarray, y: np.ndarray, splits, n_trials: int = 20, random_seed: int = 42
) -> dict:
    """Busca los mejores hiperparametros de LightGBM minimizando el MAE OOF."""

    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500, step=50),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 7, 63),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 40),
        }
        builder = lambda: LGBMRegressor(
            objective="regression_l1",
            metric="mae",
            max_depth=-1,
            random_state=random_seed,
            verbosity=-1,
            **params,
        )
        return _cv_mae(builder, X, y, splits)

    sampler = optuna.samplers.TPESampler(seed=random_seed)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params


def tune_catboost(
    X: np.ndarray, y: np.ndarray, splits, n_trials: int = 20, random_seed: int = 42
) -> dict:
    """Busca los mejores hiperparametros de CatBoost minimizando el MAE OOF."""

    def objective(trial: optuna.Trial) -> float:
        params = {
            "iterations": trial.suggest_int("iterations", 100, 500, step=50),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "depth": trial.suggest_int("depth", 4, 8),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 10.0, log=True),
        }
        builder = lambda: CatBoostRegressor(
            loss_function="MAE",
            eval_metric="MAE",
            random_seed=random_seed,
            verbose=False,
            **params,
        )
        return _cv_mae(builder, X, y, splits)

    sampler = optuna.samplers.TPESampler(seed=random_seed)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params


def build_tuned_lightgbm(best_params: dict, random_seed: int = 42) -> LGBMRegressor:
    """Construye un LightGBM Regressor usando hiperparametros ya optimizados."""
    return LGBMRegressor(
        objective="regression_l1",
        metric="mae",
        max_depth=-1,
        random_state=random_seed,
        verbosity=-1,
        **best_params,
    )


def build_tuned_catboost(best_params: dict, random_seed: int = 42) -> CatBoostRegressor:
    """Construye un CatBoost Regressor usando hiperparametros ya optimizados."""
    return CatBoostRegressor(
        loss_function="MAE",
        eval_metric="MAE",
        random_seed=random_seed,
        verbose=False,
        **best_params,
    )
