import numpy as np
import pandas as pd
import torch

from src.dl_model import BearingMLP, ACTIVATIONS, weighted_mae_loss, evaluate_activations_group_kfold


def test_all_activations_build_and_forward():
    for name in ACTIVATIONS:
        model = BearingMLP(n_features=10, activation=name)
        x = torch.randn(4, 10)
        out = model(x)
        assert out.shape == (4,)
        # salida en [0,1] por la sigmoid final (RUL fraccional valido)
        assert torch.all(out >= 0) and torch.all(out <= 1)


def test_unknown_activation_raises():
    try:
        BearingMLP(n_features=5, activation="tanh")
        assert False, "deberia levantar ValueError"
    except ValueError:
        pass


def test_weighted_mae_loss_is_nonnegative_and_zero_when_perfect():
    target = torch.tensor([0.1, 0.5, 0.9])
    perfect = weighted_mae_loss(target, target)
    assert abs(perfect.item()) < 1e-6

    off = weighted_mae_loss(target + 0.2, target)
    assert off.item() > 0


def test_weighted_mae_penalizes_low_rul_more():
    # mismo error absoluto (0.1), pero uno cerca de fin de vida (target=0.05)
    # y otro lejos (target=0.9) -- el peso debe hacer el primero mas caro.
    target_low = torch.tensor([0.05])
    target_high = torch.tensor([0.9])
    loss_low = weighted_mae_loss(target_low + 0.1, target_low)
    loss_high = weighted_mae_loss(target_high + 0.1, target_high)
    assert loss_low.item() > loss_high.item()


def test_evaluate_activations_group_kfold_runs_end_to_end_on_synthetic_data():
    # dataset sintetico pequeno, 3 grupos (como los 3 experimentos reales),
    # solo para validar que el loop de entrenamiento/evaluacion corre y
    # produce metricas validas -- no valida performance real (eso lo hace
    # el pipeline completo sobre datos reales).
    rng = np.random.default_rng(0)
    n_per_group = 30
    groups = np.repeat(["g1", "g2", "g3"], n_per_group)
    X = pd.DataFrame(rng.normal(size=(n_per_group * 3, 6)), columns=[f"f{i}" for i in range(6)])
    y = rng.uniform(0, 1, size=n_per_group * 3)

    results = evaluate_activations_group_kfold(X, y, groups, n_splits=3)

    assert set(results["model"]) == {"mlp_relu", "mlp_gelu", "mlp_swish"}
    assert (results["mae_mean"] >= 0).all()
    assert (results["mae_mean"] <= 1.5).all()


def test_validation_mask_uses_contiguous_blocks_from_every_group():
    from src.dl_model import VAL_BLOCK, validation_mask

    groups = np.repeat(["a", "b"], 200)
    mask = validation_mask(groups)
    for g in ("a", "b"):
        m = mask[groups == g]
        assert 0.05 < m.mean() < 0.3
        starts = np.flatnonzero(np.diff(np.r_[0, m.astype(int)]) == 1)
        assert all(s % VAL_BLOCK == 0 for s in starts)  # cada tramo empieza en un borde de bloque


def test_fold_training_respects_minimum_and_resumes_from_checkpoint(tmp_path, monkeypatch):
    import pytest

    from src.dl_model import _train_one_fold

    rng = np.random.default_rng(1)
    X = rng.normal(size=(120, 5)).astype(np.float32)
    y = rng.uniform(0, 1, size=120).astype(np.float32)
    kw = dict(X_val=X[90:], y_val=y[90:], epochs=40, min_epochs=40, patience=1000, return_info=True)
    ref_mae, ref = _train_one_fold(X[:90], y[:90], X[100:], y[100:], "gelu", **kw)
    assert ref["epochs_run"] == 40

    ckpt = tmp_path / "f.ckpt"
    real_step = torch.optim.Adam.step
    calls = {"n": 0}

    def flaky_step(self, *args, **k):
        calls["n"] += 1
        if calls["n"] > 25:  # corta en la epoca 26; el ultimo checkpoint es la 20
            raise KeyboardInterrupt
        return real_step(self, *args, **k)

    monkeypatch.setattr(torch.optim.Adam, "step", flaky_step)
    with pytest.raises(KeyboardInterrupt):
        _train_one_fold(X[:90], y[:90], X[100:], y[100:], "gelu", checkpoint_path=ckpt, **kw)
    monkeypatch.setattr(torch.optim.Adam, "step", real_step)

    mae, info = _train_one_fold(X[:90], y[:90], X[100:], y[100:], "gelu", checkpoint_path=ckpt, **kw)
    assert info["val_loss"] == pytest.approx(ref["val_loss"], rel=1e-5)
    assert mae == pytest.approx(ref_mae, rel=1e-5)
