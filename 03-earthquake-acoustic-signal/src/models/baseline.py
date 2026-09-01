"""Modelos baseline: Ridge, Lasso y Random Forest sobre las features tabulares.

Autor: Pablo Reyes
"""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def build_ridge_model(random_seed: int = 42):
    """Construye un pipeline de escalado + regresion Ridge."""
    return make_pipeline(StandardScaler(), Ridge(alpha=1.0, random_state=random_seed))


def build_lasso_model(random_seed: int = 42):
    """Construye un pipeline de escalado + regresion Lasso."""
    return make_pipeline(StandardScaler(), Lasso(alpha=0.01, random_state=random_seed, max_iter=5000))


def build_random_forest_model(random_seed: int = 42):
    """Construye un Random Forest Regressor con hiperparametros conservadores."""
    return RandomForestRegressor(
        n_estimators=200,
        max_depth=8,
        min_samples_leaf=2,
        random_state=random_seed,
        n_jobs=-1,
    )
