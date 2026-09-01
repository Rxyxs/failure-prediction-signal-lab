"""Comparacion de funciones de activacion (ReLU / GELU / Swish) con una loss
custom (Huber ponderada) para la CNN 1D.

Este modulo es complementario a ``src/models/cnn_model.py``: reutiliza el
mismo dataset y arquitectura general, pero permite variar la funcion de
activacion de las capas convolucionales/lineales y reemplaza ``nn.L1Loss``
por una Huber loss ponderada que da mas peso a los segmentos cercanos a la
falla (``time_to_failure`` bajo), que son los mas dificiles y los mas
relevantes desde el punto de vista de un sistema de alerta temprana.

Autor: Pablo Reyes
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from src.config import TrainingConfig
from src.models.cnn_model import SignalDataset

ACTIVATIONS: dict[str, type[nn.Module]] = {
    "relu": nn.ReLU,
    "gelu": nn.GELU,
    "swish": nn.SiLU,  # SiLU(x) = x * sigmoid(x) es la definicion estandar de Swish
}


class WeightedHuberLoss(nn.Module):
    """Huber loss (smooth L1) ponderada por cercania a la falla.

    A cada muestra se le asigna un peso en ``[min_weight, max_weight]`` que
    crece exponencialmente a medida que ``time_to_failure`` se acerca a cero,
    de forma que el modelo es penalizado con mas fuerza por errores justo
    antes de un evento de falla que por errores en regiones de calma.
    """

    def __init__(self, delta: float = 1.0, min_weight: float = 0.5, max_weight: float = 2.5, scale: float = 4.0):
        super().__init__()
        self.delta = delta
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.scale = scale

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        residual = preds - targets
        abs_residual = residual.abs()
        quadratic = torch.clamp(abs_residual, max=self.delta)
        linear = abs_residual - quadratic
        huber = 0.5 * quadratic**2 + self.delta * linear

        weights = self.min_weight + (self.max_weight - self.min_weight) * torch.exp(
            -torch.clamp(targets, min=0.0) / self.scale
        )
        return (huber * weights).mean()


class CNN1DActivation(nn.Module):
    """Variante de ``CNN1D`` con la funcion de activacion parametrizable."""

    def __init__(self, input_length: int, activation: str = "relu"):
        super().__init__()
        if activation not in ACTIVATIONS:
            raise ValueError(f"Activacion desconocida: {activation}. Opciones: {list(ACTIVATIONS)}")
        act_cls = ACTIVATIONS[activation]
        self.activation_name = activation
        self.features = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=15, stride=2, padding=7),
            nn.BatchNorm1d(16),
            act_cls(),
            nn.MaxPool1d(2),
            nn.Conv1d(16, 32, kernel_size=9, stride=2, padding=4),
            nn.BatchNorm1d(32),
            act_cls(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(64),
            act_cls(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.regressor = nn.Sequential(
            nn.Linear(64, 32),
            act_cls(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.squeeze(-1)
        return self.regressor(x)


@dataclass
class ActivationRunResult:
    """Resultado de entrenar la CNN con una activacion y loss dadas."""

    activation: str
    val_mae: float
    epoch_losses: list[float] = field(default_factory=list)


def train_with_activation(
    train_signals: np.ndarray,
    train_targets: np.ndarray,
    val_signals: np.ndarray,
    val_targets: np.ndarray,
    activation: str,
    config: TrainingConfig,
    loss_fn: nn.Module | None = None,
    device: str = "cpu",
) -> ActivationRunResult:
    """Entrena ``CNN1DActivation`` con la activacion indicada y una loss custom."""
    input_length = train_signals.shape[1]
    model = CNN1DActivation(input_length=input_length, activation=activation).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.cnn_learning_rate)
    criterion = loss_fn if loss_fn is not None else WeightedHuberLoss()

    train_loader = DataLoader(
        SignalDataset(train_signals, train_targets),
        batch_size=config.cnn_batch_size,
        shuffle=True,
    )

    epoch_losses: list[float] = []
    model.train()
    for _epoch in range(config.cnn_epochs):
        running_loss = 0.0
        n_batches = 0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            preds = model(batch_x)
            loss = criterion(preds, batch_y)
            loss.backward()
            optimizer.step()
            running_loss += float(loss.item())
            n_batches += 1
        epoch_losses.append(running_loss / max(n_batches, 1))

    model.eval()
    with torch.no_grad():
        val_x = torch.tensor(val_signals, dtype=torch.float32).unsqueeze(1).to(device)
        val_preds = model(val_x).squeeze(-1).cpu().numpy()

    mae = float(np.mean(np.abs(val_preds - val_targets)))
    return ActivationRunResult(activation=activation, val_mae=mae, epoch_losses=epoch_losses)


def run_activation_comparison(
    train_signals: np.ndarray,
    train_targets: np.ndarray,
    val_signals: np.ndarray,
    val_targets: np.ndarray,
    config: TrainingConfig,
    activations: tuple[str, ...] = ("relu", "gelu", "swish"),
    device: str = "cpu",
) -> dict[str, ActivationRunResult]:
    """Entrena la CNN con cada activacion en ``activations`` y retorna sus resultados."""
    results: dict[str, ActivationRunResult] = {}
    for activation in activations:
        # Semilla fija por activacion para que la comparacion no dependa de la
        # inicializacion aleatoria de cada corrida.
        torch.manual_seed(config.random_seed)
        results[activation] = train_with_activation(
            train_signals, train_targets, val_signals, val_targets, activation, config, device=device
        )
    return results
