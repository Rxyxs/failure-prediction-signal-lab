"""CNN 1D en PyTorch para regresion directa sobre la forma de onda cruda.

La red toma un segmento (submuestreado para mantener el entrenamiento en
tiempos razonables) y predice ``time_to_failure`` directamente a partir de la
senal acustica, sin pasar por features manuales.

Autor: Pablo Reyes
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from src.config import TrainingConfig


class SignalDataset(Dataset):
    """Dataset de PyTorch para segmentos de senal acustica submuestreados."""

    def __init__(self, signals: np.ndarray, targets: np.ndarray):
        self.signals = torch.tensor(signals, dtype=torch.float32).unsqueeze(1)  # (N, 1, L)
        self.targets = torch.tensor(targets, dtype=torch.float32).unsqueeze(1)  # (N, 1)

    def __len__(self) -> int:
        return len(self.signals)

    def __getitem__(self, idx):
        return self.signals[idx], self.targets[idx]


class CNN1D(nn.Module):
    """Arquitectura convolucional 1D compacta para regresion de secuencias."""

    def __init__(self, input_length: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=15, stride=2, padding=7),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(16, 32, kernel_size=9, stride=2, padding=4),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.regressor = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.squeeze(-1)
        return self.regressor(x)


def subsample_signal(signal: np.ndarray, target_length: int) -> np.ndarray:
    """Submuestrea una senal larga a ``target_length`` puntos via decimacion uniforme."""
    if len(signal) <= target_length:
        return signal
    indices = np.linspace(0, len(signal) - 1, target_length).astype(int)
    return signal[indices]


def train_cnn_model(
    train_signals: np.ndarray,
    train_targets: np.ndarray,
    val_signals: np.ndarray,
    val_targets: np.ndarray,
    config: TrainingConfig,
    device: str = "cpu",
    checkpoint_path: Path | None = None,
) -> tuple[CNN1D, float]:
    """Entrena la CNN 1D (``fit_with_early_stopping``) y retorna el modelo junto
    con el MAE sobre ``val_signals``, que el entrenamiento nunca mira."""
    torch.manual_seed(config.random_seed)
    input_length = train_signals.shape[1]
    model = CNN1D(input_length=input_length).to(device)
    fit_with_early_stopping(model, nn.L1Loss(), train_signals, train_targets, config, device=device,
                            checkpoint_path=checkpoint_path)

    with torch.no_grad():
        val_x = torch.tensor(val_signals, dtype=torch.float32).unsqueeze(1).to(device)
        val_preds = model(val_x).squeeze(-1).cpu().numpy()

    mae = float(np.mean(np.abs(val_preds - val_targets)))
    return model, mae


def predict_cnn(model: CNN1D, signals: np.ndarray, device: str = "cpu") -> np.ndarray:
    """Genera predicciones con una CNN 1D ya entrenada."""
    model.eval()
    with torch.no_grad():
        x = torch.tensor(signals, dtype=torch.float32).unsqueeze(1).to(device)
        preds = model(x).squeeze(-1).cpu().numpy()
    return preds


def _fingerprint(*arrays: np.ndarray, **config) -> str:
    h = hashlib.sha256()
    for a in arrays:
        h.update(np.ascontiguousarray(a, dtype=np.float32).tobytes())
    h.update(json.dumps(config, sort_keys=True, default=str).encode())
    return h.hexdigest()[:16]


def fit_with_early_stopping(
    model: nn.Module,
    criterion: nn.Module,
    train_signals: np.ndarray,
    train_targets: np.ndarray,
    config: TrainingConfig,
    device: str = "cpu",
    checkpoint_path: Path | None = None,
) -> dict:
    """Entrena ``model`` con el protocolo comun del proyecto.

    El tramo final (``config.cnn_val_fraction``) de los segmentos de
    entrenamiento, en orden temporal, se reserva para early stopping: el
    conjunto de evaluacion que recibe quien llama nunca se mira aqui. Al menos
    ``cnn_min_epochs`` epocas (o ``cnn_epochs`` si es menor); desde ahi corta
    tras ``cnn_patience`` epocas sin mejora en la loss de validacion y
    restaura la mejor epoca. Con ``checkpoint_path`` el estado completo se
    guarda cada epoca y una corrida interrumpida retoma donde quedo.
    Devuelve la historia por epoca."""
    n_val = max(1, int(len(train_signals) * config.cnn_val_fraction))
    fit_x, fit_y = train_signals[:-n_val], train_targets[:-n_val]
    val_x = torch.tensor(train_signals[-n_val:], dtype=torch.float32).unsqueeze(1).to(device)
    val_y = torch.tensor(train_targets[-n_val:], dtype=torch.float32).unsqueeze(1).to(device)  # (N, 1), como en SignalDataset

    optimizer = torch.optim.Adam(model.parameters(), lr=config.cnn_learning_rate)
    generator = torch.Generator()
    loader = DataLoader(SignalDataset(fit_x, fit_y), batch_size=config.cnn_batch_size, shuffle=True,
                        generator=generator)

    fp = _fingerprint(train_signals, train_targets, model=type(model).__name__,
                      act=getattr(model, "activation_name", None), loss=type(criterion).__name__,
                      epochs=config.cnn_epochs, min_epochs=config.cnn_min_epochs, patience=config.cnn_patience,
                      lr=config.cnn_learning_rate, batch=config.cnn_batch_size, seed=config.random_seed,
                      val_fraction=config.cnn_val_fraction)
    state = {"epoch": 0, "train_loss": [], "val_loss": [], "best_val": float("inf"), "best_epoch": 0,
             "best_state": None, "bad_epochs": 0, "done": False}
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

    while not state["done"] and state["epoch"] < config.cnn_epochs:
        generator.manual_seed(config.random_seed + state["epoch"])  # mismo orden de batches al retomar
        torch.manual_seed(config.random_seed + 10_000 + state["epoch"])
        model.train()
        running, n_batches = 0.0, 0
        for batch_x, batch_y in loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(batch_x), batch_y)
            loss.backward()
            optimizer.step()
            running += float(loss.item())
            n_batches += 1
        state["epoch"] += 1
        state["train_loss"].append(running / max(n_batches, 1))

        model.eval()
        with torch.no_grad():
            val_loss = float(criterion(model(val_x), val_y).item())
        state["val_loss"].append(val_loss)
        if val_loss < state["best_val"] - 1e-7:
            state.update(best_val=val_loss, best_epoch=state["epoch"], bad_epochs=0,
                         best_state=copy.deepcopy(model.state_dict()))
        else:
            state["bad_epochs"] += 1
        if state["epoch"] >= min(config.cnn_min_epochs, config.cnn_epochs) and state["bad_epochs"] >= config.cnn_patience:
            state["done"] = True
        save_checkpoint()

    if state["best_state"] is not None:
        model.load_state_dict(state["best_state"])
    state["done"] = True
    save_checkpoint()
    model.eval()
    return {"train_loss": state["train_loss"], "val_loss": state["val_loss"],
            "best_epoch": state["best_epoch"], "epochs_run": state["epoch"]}
