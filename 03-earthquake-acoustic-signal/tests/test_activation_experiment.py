"""Pruebas del modulo de comparacion de activaciones (ReLU/GELU/Swish) y de la
loss custom (Huber ponderada) para la CNN 1D.

Autor: Pablo Reyes
"""
from __future__ import annotations

import numpy as np
import torch

from src.config import TrainingConfig
from src.models.activation_experiment import (
    ACTIVATIONS,
    CNN1DActivation,
    WeightedHuberLoss,
    run_activation_comparison,
    train_with_activation,
)
from src.viz.plots import plot_activation_comparison, plot_activation_loss_curves


def test_weighted_huber_loss_is_nonnegative_and_zero_at_perfect_fit():
    criterion = WeightedHuberLoss()
    targets = torch.tensor([[1.0], [3.0], [5.0]])
    loss_perfect = criterion(targets, targets)
    assert loss_perfect.item() == 0.0

    preds = targets + 2.0
    loss_off = criterion(preds, targets)
    assert loss_off.item() > 0.0


def test_weighted_huber_loss_weights_near_failure_more():
    """Un mismo error absoluto debe pesar mas cuando time_to_failure es bajo."""
    criterion = WeightedHuberLoss(delta=1.0, min_weight=0.5, max_weight=2.5, scale=4.0)
    error = 3.0  # mayor al delta, cae en el tramo lineal donde el peso es mas visible

    target_near_failure = torch.tensor([[0.0]])
    target_far_from_failure = torch.tensor([[20.0]])

    loss_near = criterion(target_near_failure + error, target_near_failure)
    loss_far = criterion(target_far_from_failure + error, target_far_from_failure)

    assert loss_near.item() > loss_far.item()


def test_cnn1d_activation_forward_shapes_for_each_activation():
    batch_size, length = 4, 512
    x = torch.randn(batch_size, 1, length)
    for name in ACTIVATIONS:
        model = CNN1DActivation(input_length=length, activation=name)
        out = model(x)
        assert out.shape == (batch_size, 1)


def test_cnn1d_activation_rejects_unknown_activation():
    try:
        CNN1DActivation(input_length=128, activation="tanh")
        assert False, "se esperaba ValueError para una activacion desconocida"
    except ValueError:
        pass


def _toy_dataset(n_train=6, n_val=2, length=256, seed=0):
    rng = np.random.default_rng(seed)
    train_signals = rng.normal(size=(n_train, length)).astype(np.float32)
    train_targets = rng.uniform(0, 10, size=n_train).astype(np.float32)
    val_signals = rng.normal(size=(n_val, length)).astype(np.float32)
    val_targets = rng.uniform(0, 10, size=n_val).astype(np.float32)
    return train_signals, train_targets, val_signals, val_targets


def test_train_with_activation_returns_mae_and_epoch_losses():
    train_signals, train_targets, val_signals, val_targets = _toy_dataset()
    config = TrainingConfig(cnn_epochs=2, cnn_batch_size=2)

    result = train_with_activation(
        train_signals, train_targets, val_signals, val_targets, activation="gelu", config=config
    )

    assert result.activation == "gelu"
    assert result.val_mae >= 0.0
    assert len(result.epoch_losses) == config.cnn_epochs
    assert all(loss >= 0.0 for loss in result.epoch_losses)


def test_run_activation_comparison_covers_all_default_activations():
    train_signals, train_targets, val_signals, val_targets = _toy_dataset()
    config = TrainingConfig(cnn_epochs=1, cnn_batch_size=2)

    results = run_activation_comparison(train_signals, train_targets, val_signals, val_targets, config)

    assert set(results.keys()) == {"relu", "gelu", "swish"}
    for result in results.values():
        assert result.val_mae >= 0.0
        assert len(result.epoch_losses) == 1


def _assert_png(path) -> None:
    assert path.exists()
    assert path.stat().st_size > 0
    with open(path, "rb") as fh:
        assert fh.read(8) == b"\x89PNG\r\n\x1a\n"


def test_plot_activation_comparison(tmp_path):
    out = plot_activation_comparison({"relu": 2.4, "gelu": 2.5, "swish": 2.7}, tmp_path / "activation_comparison.png")
    _assert_png(out)


def test_plot_activation_loss_curves(tmp_path):
    out = plot_activation_loss_curves(
        {"relu": [3.0, 2.5, 2.1], "gelu": [3.1, 2.4, 2.0], "swish": [3.2, 2.6, 2.3]},
        tmp_path / "activation_loss_curves.png",
    )
    _assert_png(out)
