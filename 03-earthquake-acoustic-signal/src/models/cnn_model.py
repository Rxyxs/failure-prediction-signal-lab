"""CNN 1D en PyTorch para regresion directa sobre la forma de onda cruda.

La red toma un segmento (submuestreado para mantener el entrenamiento en
tiempos razonables) y predice ``time_to_failure`` directamente a partir de la
senal acustica, sin pasar por features manuales.

Autor: Pablo Reyes
"""
from __future__ import annotations

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
) -> tuple[CNN1D, float]:
    """Entrena la CNN 1D y retorna el modelo entrenado junto con el MAE de validacion."""
    input_length = train_signals.shape[1]
    model = CNN1D(input_length=input_length).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.cnn_learning_rate)
    criterion = nn.L1Loss()

    train_loader = DataLoader(
        SignalDataset(train_signals, train_targets),
        batch_size=config.cnn_batch_size,
        shuffle=True,
    )

    model.train()
    for _epoch in range(config.cnn_epochs):
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            preds = model(batch_x)
            loss = criterion(preds, batch_y)
            loss.backward()
            optimizer.step()

    model.eval()
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
