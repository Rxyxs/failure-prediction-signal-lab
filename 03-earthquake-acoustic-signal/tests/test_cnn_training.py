import numpy as np
import pytest
import torch
from torch import nn

from src.config import TrainingConfig
from src.models.cnn_model import CNN1D, fit_with_early_stopping


def _data(n=24, length=300, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, length)).astype(np.float32)
    y = np.abs(x[:, :50].mean(axis=1) * 5 + rng.normal(scale=0.1, size=n)).astype(np.float32)
    return x, y


def test_never_stops_before_minimum_and_restores_best_epoch():
    x, y = _data()
    cfg = TrainingConfig(cnn_epochs=40, cnn_min_epochs=12, cnn_patience=2, cnn_batch_size=4)
    torch.manual_seed(0)
    model = CNN1D(input_length=x.shape[1])
    hist = fit_with_early_stopping(model, nn.L1Loss(), x, y, cfg)
    assert hist["epochs_run"] >= 12
    assert hist["best_epoch"] == int(np.argmin(hist["val_loss"])) + 1


def test_interrupted_run_resumes_from_checkpoint(tmp_path, monkeypatch):
    x, y = _data(seed=1)
    cfg = TrainingConfig(cnn_epochs=5, cnn_min_epochs=5, cnn_patience=100, cnn_batch_size=4)
    torch.manual_seed(0)
    ref_model = CNN1D(input_length=x.shape[1])
    ref = fit_with_early_stopping(ref_model, nn.L1Loss(), x, y, cfg)

    ckpt = tmp_path / "cnn.ckpt"
    real_step = torch.optim.Adam.step
    calls = {"n": 0}
    n_batches = -(-(len(x) - max(1, int(len(x) * cfg.cnn_val_fraction))) // cfg.cnn_batch_size)

    def flaky_step(self, *args, **kw):
        calls["n"] += 1
        if calls["n"] > n_batches * 2:  # corta durante la epoca 3
            raise KeyboardInterrupt
        return real_step(self, *args, **kw)

    monkeypatch.setattr(torch.optim.Adam, "step", flaky_step)
    torch.manual_seed(0)
    with pytest.raises(KeyboardInterrupt):
        fit_with_early_stopping(CNN1D(input_length=x.shape[1]), nn.L1Loss(), x, y, cfg, checkpoint_path=ckpt)
    monkeypatch.setattr(torch.optim.Adam, "step", real_step)

    torch.manual_seed(0)
    resumed_model = CNN1D(input_length=x.shape[1])
    hist = fit_with_early_stopping(resumed_model, nn.L1Loss(), x, y, cfg, checkpoint_path=ckpt)
    assert hist["val_loss"] == pytest.approx(ref["val_loss"], rel=1e-5)
    for a, b in zip(ref_model.state_dict().values(), resumed_model.state_dict().values()):
        assert torch.allclose(a.float(), b.float(), atol=1e-6)
