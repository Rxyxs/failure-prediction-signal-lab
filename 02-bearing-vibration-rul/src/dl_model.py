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

import copy
import hashlib
import json
from pathlib import Path

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


MIN_EPOCHS = 100   # ninguna red se detiene antes de 100 epocas
MAX_EPOCHS = 2000  # epocas full-batch: cada una es un solo paso de gradiente
PATIENCE = 50
VAL_FRACTION = 0.15
VAL_BLOCK = 20     # snapshots consecutivos por bloque de validacion
CHECKPOINT_EVERY = 10


def validation_mask(groups: np.ndarray, seed: int = RANDOM_STATE) -> np.ndarray:
    """Validacion para el early stopping, tomada de los rigs de entrenamiento:
    cada rig se corta en bloques de ``VAL_BLOCK`` snapshots consecutivos y un
    ``VAL_FRACTION`` de los bloques va a validacion. Bloques contiguos y no
    snapshots sueltos, porque dos snapshots vecinos (10 minutos de diferencia)
    son casi identicos y una validacion aleatoria fila a fila seria optimista."""
    rng = np.random.default_rng(seed)
    mask = np.zeros(len(groups), dtype=bool)
    for g in pd.unique(groups):
        idx = np.flatnonzero(groups == g)
        blocks = [idx[i:i + VAL_BLOCK] for i in range(0, len(idx), VAL_BLOCK)]
        n_val = max(1, int(round(len(blocks) * VAL_FRACTION)))
        for b in rng.choice(len(blocks), size=n_val, replace=False):
            mask[blocks[b]] = True
    return mask


def _fingerprint(*arrays: np.ndarray, **config) -> str:
    h = hashlib.sha256()
    for a in arrays:
        h.update(np.ascontiguousarray(a, dtype=np.float32).tobytes())
    h.update(json.dumps(config, sort_keys=True).encode())
    return h.hexdigest()[:16]


def _train_one_fold(
    X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray,
    activation: str, epochs: int = MAX_EPOCHS, lr: float = 1e-3,
    X_val: np.ndarray | None = None, y_val: np.ndarray | None = None,
    min_epochs: int = MIN_EPOCHS, patience: int = PATIENCE,
    checkpoint_path: Path | None = None, return_info: bool = False,
):
    """Entrena una red en un fold. Con validacion: al menos ``min_epochs``
    epocas (o ``epochs`` si es menor) y desde ahi early stopping tras
    ``patience`` epocas sin mejora, restaurando la mejor epoca. El estado se
    guarda cada ``CHECKPOINT_EVERY`` epocas para retomar una corrida
    interrumpida. Devuelve el MAE sobre el rig de test."""
    torch.manual_seed(RANDOM_STATE)
    model = BearingMLP(n_features=X_train.shape[1], activation=activation)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    xt = torch.tensor(X_train, dtype=torch.float32)
    yt = torch.tensor(y_train, dtype=torch.float32)
    xv = torch.tensor(X_test, dtype=torch.float32)
    yv = torch.tensor(y_test, dtype=torch.float32)
    has_val = X_val is not None and y_val is not None and len(X_val) > 0
    if has_val:
        xval = torch.tensor(X_val, dtype=torch.float32)
        yval = torch.tensor(y_val, dtype=torch.float32)

    fp = _fingerprint(X_train, y_train, *((X_val, y_val) if has_val else ()), activation=activation,
                      epochs=epochs, lr=lr, min_epochs=min_epochs, patience=patience)
    state = {"epoch": 0, "val_loss": [], "best_val": float("inf"), "best_epoch": 0, "best_state": None,
             "bad_epochs": 0, "done": False}
    if checkpoint_path is not None and Path(checkpoint_path).exists():
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if ckpt.get("fingerprint") == fp:
            model.load_state_dict(ckpt["model"])
            optimizer.load_state_dict(ckpt["optimizer"])
            state = ckpt["state"]

    def save_checkpoint() -> None:
        if checkpoint_path is None:
            return
        path = Path(checkpoint_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        torch.save({"fingerprint": fp, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "state": state}, tmp)
        tmp.replace(path)

    while not state["done"] and state["epoch"] < epochs:
        torch.manual_seed(RANDOM_STATE + state["epoch"])  # mascaras de dropout reproducibles al retomar
        model.train()
        optimizer.zero_grad()
        loss = weighted_mae_loss(model(xt), yt)
        loss.backward()
        optimizer.step()
        state["epoch"] += 1
        if has_val:
            model.eval()
            with torch.no_grad():
                val_loss = weighted_mae_loss(model(xval), yval).item()
            state["val_loss"].append(val_loss)
            if val_loss < state["best_val"] - 1e-7:
                state.update(best_val=val_loss, best_epoch=state["epoch"], bad_epochs=0,
                             best_state=copy.deepcopy(model.state_dict()))
            else:
                state["bad_epochs"] += 1
            if state["epoch"] >= min(min_epochs, epochs) and state["bad_epochs"] >= patience:
                state["done"] = True
        if state["epoch"] % CHECKPOINT_EVERY == 0:
            save_checkpoint()

    if has_val and state["best_state"] is not None:
        model.load_state_dict(state["best_state"])
    state["done"] = True
    save_checkpoint()

    model.eval()
    with torch.no_grad():
        pred_val = model(xv)
        mae = torch.mean(torch.abs(pred_val - yv)).item()
    if return_info:
        return mae, {"epochs_run": state["epoch"], "best_epoch": state["best_epoch"] if has_val else state["epoch"],
                     "val_loss": state["val_loss"]}
    return mae


def evaluate_activations_group_kfold(
    X: pd.DataFrame, y: np.ndarray, groups: np.ndarray, n_splits: int = 3,
    checkpoint_dir: Path | None = None, epochs: int = MAX_EPOCHS, min_epochs: int = MIN_EPOCHS,
) -> pd.DataFrame:
    """Mismo protocolo GroupKFold leave-one-experiment-out que src.modeling,
    comparando ReLU vs GELU vs Swish con la misma arquitectura/loss. Dentro de
    cada fold, la validacion del early stopping sale de los rigs de
    entrenamiento (``validation_mask``); el rig de test solo se usa para el MAE."""
    gkf = GroupKFold(n_splits=n_splits)
    results = []

    for activation in ACTIVATIONS:
        fold_maes, best_epochs = [], []
        for fold, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups), start=1):
            val_in_train = validation_mask(groups[train_idx])
            fit_idx, val_idx = train_idx[~val_in_train], train_idx[val_in_train]
            scaler = StandardScaler()
            X_fit = scaler.fit_transform(X.iloc[fit_idx])
            X_val = scaler.transform(X.iloc[val_idx])
            X_test = scaler.transform(X.iloc[test_idx])
            ckpt = None if checkpoint_dir is None else Path(checkpoint_dir) / f"mlp_{activation}_fold{fold}.ckpt"
            mae, info = _train_one_fold(X_fit, y[fit_idx], X_test, y[test_idx], activation=activation,
                                        X_val=X_val, y_val=y[val_idx], epochs=epochs, min_epochs=min_epochs,
                                        checkpoint_path=ckpt, return_info=True)
            fold_maes.append(mae)
            best_epochs.append(info["best_epoch"])
            print(f"  [mlp-{activation}] fold {fold} (held-out group={groups[test_idx][0]}): MAE={mae:.4f} "
                  f"(mejor epoca {info['best_epoch']} de {info['epochs_run']})")
        results.append({
            "model": f"mlp_{activation}",
            "mae_mean": float(np.mean(fold_maes)),
            "mae_std": float(np.std(fold_maes)),
            "best_epochs": ",".join(str(e) for e in best_epochs),
        })

    return pd.DataFrame(results).sort_values("mae_mean")
