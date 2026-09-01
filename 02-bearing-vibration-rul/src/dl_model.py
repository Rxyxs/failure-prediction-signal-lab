"""Tercer enfoque de modelado: MLP en PyTorch sobre las mismas features
espectrales/temporales que Ridge/RF/LightGBM/CatBoost, con comparacion de
funciones de activacion (ReLU / GELU / Swish) y una loss custom.

Complementario a src/modeling.py, no lo reemplaza: mismo protocolo de
validacion (GroupKFold leave-one-experiment-out, 3 folds = 3 rigs fisicos
independientes), mismo target (RUL fraccional [0,1]), para que las metricas
sean directamente comparables en la tabla del README.

    python -m src.dl_model
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from torch import nn

RANDOM_STATE = 42
torch.manual_seed(RANDOM_STATE)

ACTIVATIONS = {
    "relu": nn.ReLU,
    "gelu": nn.GELU,
    "swish": nn.SiLU,  # Swish == SiLU (x * sigmoid(x))
}


class BearingMLP(nn.Module):
    """MLP simple: capas densas + dropout, activacion configurable, salida
    escalar en [0,1] via sigmoid (RUL fraccional)."""

    def __init__(self, n_features: int, activation: str = "relu", hidden: tuple[int, ...] = (64, 32)):
        super().__init__()
        if activation not in ACTIVATIONS:
            raise ValueError(f"activacion desconocida: {activation!r}, opciones: {list(ACTIVATIONS)}")
        act_cls = ACTIVATIONS[activation]

        layers: list[nn.Module] = []
        in_dim = n_features
        for h in hidden:
            layers += [nn.Linear(in_dim, h), act_cls(), nn.Dropout(0.15)]
            in_dim = h
        layers += [nn.Linear(in_dim, 1), nn.Sigmoid()]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def weighted_mae_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Loss custom: MAE ponderado que penaliza mas el error cerca del fin de
    vida util (target -> 0), donde una prediccion optimista es mas costosa
    en mantenimiento predictivo real que sobreestimar RUL al inicio de vida."""
    weight = 1.0 + 2.0 * (1.0 - target)  # peso en [1, 3], mayor cerca de target=0
    return torch.mean(weight * torch.abs(pred - target))


def _train_one_fold(
    X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray,
    activation: str, epochs: int = 80, lr: float = 1e-3,
) -> float:
    torch.manual_seed(RANDOM_STATE)
    model = BearingMLP(n_features=X_train.shape[1], activation=activation)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    xt = torch.tensor(X_train, dtype=torch.float32)
    yt = torch.tensor(y_train, dtype=torch.float32)
    xv = torch.tensor(X_test, dtype=torch.float32)
    yv = torch.tensor(y_test, dtype=torch.float32)

    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        pred = model(xt)
        loss = weighted_mae_loss(pred, yt)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        pred_val = model(xv)
        mae = torch.mean(torch.abs(pred_val - yv)).item()
    return mae


def evaluate_activations_group_kfold(
    X: pd.DataFrame, y: np.ndarray, groups: np.ndarray, n_splits: int = 3,
) -> pd.DataFrame:
    """Mismo protocolo GroupKFold leave-one-experiment-out que src.modeling,
    comparando ReLU vs GELU vs Swish con la misma arquitectura/loss/epocas."""
    gkf = GroupKFold(n_splits=n_splits)
    results = []

    for activation in ACTIVATIONS:
        fold_maes = []
        for fold, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups), start=1):
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X.iloc[train_idx])
            X_test = scaler.transform(X.iloc[test_idx])
            mae = _train_one_fold(X_train, y[train_idx], X_test, y[test_idx], activation=activation)
            fold_maes.append(mae)
            print(f"  [mlp-{activation}] fold {fold} (held-out group={groups[test_idx][0]}): MAE={mae:.4f}")
        results.append({
            "model": f"mlp_{activation}",
            "mae_mean": float(np.mean(fold_maes)),
            "mae_std": float(np.std(fold_maes)),
        })

    return pd.DataFrame(results).sort_values("mae_mean")
