"""Pruebas del modulo de visualizacion: cada funcion debe producir un PNG valido.

Autor: Pablo Reyes
"""
from __future__ import annotations

import numpy as np

from src.viz.plots import (
    plot_cv_mae_by_fold,
    plot_feature_importance,
    plot_mae_comparison,
    plot_prediction_vs_actual,
    plot_signal_vs_ttf,
    plot_spectrogram,
    plot_ttf_distribution,
)


def _assert_png(path) -> None:
    assert path.exists()
    assert path.stat().st_size > 0
    with open(path, "rb") as fh:
        assert fh.read(8) == b"\x89PNG\r\n\x1a\n"


def test_plot_signal_vs_ttf(tmp_path):
    rng = np.random.default_rng(0)
    signal = rng.normal(size=5000)
    ttf = np.linspace(10, 0, 5000)
    out = plot_signal_vs_ttf(signal, ttf, tmp_path / "signal_vs_ttf.png", n_points=1000)
    _assert_png(out)


def test_plot_ttf_distribution(tmp_path):
    rng = np.random.default_rng(0)
    ttf = rng.uniform(0, 10, size=500)
    out = plot_ttf_distribution(ttf, tmp_path / "ttf_dist.png")
    _assert_png(out)


def test_plot_spectrogram(tmp_path):
    rng = np.random.default_rng(0)
    signal = rng.normal(size=8000)
    out = plot_spectrogram(signal, tmp_path / "spec.png", nperseg=256)
    _assert_png(out)


def test_plot_mae_comparison(tmp_path):
    mae_by_model = {"ridge": 3.1, "lightgbm": 2.5, "ensemble": 2.0}
    out = plot_mae_comparison(mae_by_model, tmp_path / "mae.png")
    _assert_png(out)


def test_plot_prediction_vs_actual(tmp_path):
    rng = np.random.default_rng(0)
    y_true = rng.uniform(0, 10, size=100)
    preds = {"model_a": y_true + rng.normal(0, 0.5, size=100)}
    out = plot_prediction_vs_actual(y_true, preds, tmp_path / "pred_vs_actual.png")
    _assert_png(out)


def test_plot_feature_importance(tmp_path):
    feature_names = [f"f{i}" for i in range(30)]
    rng = np.random.default_rng(0)
    importances = rng.uniform(0, 1, size=30)
    out = plot_feature_importance(feature_names, importances, tmp_path / "fi.png", top_n=20)
    _assert_png(out)


def test_plot_cv_mae_by_fold(tmp_path):
    fold_scores = {"lightgbm": [3.1, 2.9, 3.0], "random_forest": [1.6, 1.5, 1.7]}
    out = plot_cv_mae_by_fold(fold_scores, tmp_path / "cv.png")
    _assert_png(out)
