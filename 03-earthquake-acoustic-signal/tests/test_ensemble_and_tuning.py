"""Pruebas del ensamble con pesos optimizados (NNLS) y de la optimizacion de hiperparametros.

Autor: Pablo Reyes
"""
from __future__ import annotations

import numpy as np

from src.models.cross_validation import get_group_kfold_splits
from src.models.ensemble import (
    compute_inverse_mae_weights,
    compute_optimized_weights,
    weighted_ensemble_predict,
)
from src.models.tuning import build_tuned_lightgbm, tune_lightgbm


def test_compute_inverse_mae_weights_sums_to_one():
    mae_by_model = {"a": 1.0, "b": 2.0, "c": 4.0}
    weights = compute_inverse_mae_weights(mae_by_model)
    assert abs(sum(weights.values()) - 1.0) < 1e-9
    assert weights["a"] > weights["b"] > weights["c"]


def test_compute_optimized_weights_are_non_negative_and_sum_to_one():
    rng = np.random.default_rng(0)
    y = rng.normal(size=200)
    oof_by_model = {
        "good": y + rng.normal(0, 0.1, size=200),
        "bad": rng.normal(size=200),
    }
    weights = compute_optimized_weights(oof_by_model, y)
    assert abs(sum(weights.values()) - 1.0) < 1e-6
    assert all(w >= -1e-9 for w in weights.values())
    # El modelo casi perfecto debe recibir mucho mas peso que el ruidoso.
    assert weights["good"] > weights["bad"]


def test_optimized_ensemble_outperforms_worst_single_model():
    rng = np.random.default_rng(2)
    y = rng.normal(size=300)
    oof_by_model = {
        "good": y + rng.normal(0, 0.2, size=300),
        "noisy": y + rng.normal(0, 3.0, size=300),
    }
    weights = compute_optimized_weights(oof_by_model, y)
    ensemble_pred = weighted_ensemble_predict(oof_by_model, weights)

    mae_ensemble = float(np.mean(np.abs(ensemble_pred - y)))
    mae_noisy = float(np.mean(np.abs(oof_by_model["noisy"] - y)))
    assert mae_ensemble < mae_noisy


def test_tune_lightgbm_returns_usable_params():
    rng = np.random.default_rng(3)
    n_samples, n_features = 120, 10
    X = rng.normal(size=(n_samples, n_features))
    y = X[:, 0] * 2.0 + rng.normal(0, 0.1, size=n_samples)
    splits, _ = get_group_kfold_splits(n_samples, n_splits=3)

    best_params = tune_lightgbm(X, y, splits, n_trials=3, random_seed=0)
    assert "n_estimators" in best_params
    assert "learning_rate" in best_params

    model = build_tuned_lightgbm(best_params, random_seed=0)
    model.fit(X, y)
    preds = model.predict(X)
    assert preds.shape == y.shape
