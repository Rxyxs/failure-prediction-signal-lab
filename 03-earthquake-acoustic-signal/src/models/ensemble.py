"""Ensamble de predicciones de multiples modelos.

Se ofrecen dos estrategias:

1. Pesos por inverso del MAE (``compute_inverse_mae_weights``): heuristica
   simple donde los modelos mas precisos dominan la prediccion final.
2. Pesos optimizados por minimos cuadrados no negativos (NNLS) sobre las
   predicciones out-of-fold (``compute_optimized_weights``): en vez de una
   heuristica, se resuelve directamente el problema de encontrar los pesos
   ``w >= 0`` que minimizan ``||y - sum_i w_i * oof_i||^2`` usando
   ``scipy.optimize.nnls``, y luego se normalizan para que sumen 1. Esto le
   permite al ensamble aprender, por ejemplo, a asignar peso cero a un modelo
   que no aporta senal marginal (como la CNN 1D quedando dominada por los
   modelos tabulares), en vez de depender de una regla fija basada solo en el
   MAE individual de cada modelo.

Autor: Pablo Reyes
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import nnls


def compute_inverse_mae_weights(mae_by_model: dict[str, float]) -> dict[str, float]:
    """Calcula pesos normalizados inversamente proporcionales al MAE de cada modelo."""
    inverse = {name: 1.0 / max(mae, 1e-6) for name, mae in mae_by_model.items()}
    total = sum(inverse.values())
    return {name: value / total for name, value in inverse.items()}


def compute_optimized_weights(
    oof_by_model: dict[str, np.ndarray], y: np.ndarray
) -> dict[str, float]:
    """Resuelve pesos no-negativos que minimizan el error cuadratico del ensamble sobre OOF.

    Si la optimizacion NNLS entrega un vector de pesos todo-cero (caso
    degenerado con predicciones muy ruidosas), recae en pesos uniformes para
    evitar un ensamble invalido.
    """
    names = list(oof_by_model.keys())
    A = np.stack([oof_by_model[name] for name in names], axis=1)  # (n_samples, n_models)
    weights_vector, _residual = nnls(A, y)

    total = weights_vector.sum()
    if total <= 1e-12:
        uniform = 1.0 / len(names)
        return {name: uniform for name in names}

    weights_vector = weights_vector / total
    return {name: float(w) for name, w in zip(names, weights_vector)}


def weighted_ensemble_predict(
    predictions_by_model: dict[str, np.ndarray], weights: dict[str, float]
) -> np.ndarray:
    """Combina las predicciones de cada modelo usando los pesos dados."""
    names = list(predictions_by_model.keys())
    stacked = np.stack([predictions_by_model[name] for name in names], axis=0)
    weight_vector = np.array([weights[name] for name in names]).reshape(-1, 1)
    return (stacked * weight_vector).sum(axis=0)
