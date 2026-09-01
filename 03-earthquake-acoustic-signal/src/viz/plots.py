"""Funciones de graficacion para el analisis del proyecto.

Cada funcion recibe datos ya calculados (senal, reportes de MAE, modelos
entrenados, etc.) y guarda una figura PNG en el directorio indicado. Se usa
matplotlib puro (sin estado global compartido entre funciones) para que cada
grafico pueda generarse y probarse de forma independiente.

Autor: Pablo Reyes
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # backend no interactivo, apto para scripts/CI

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.animation import FuncAnimation
from scipy.signal import spectrogram

plt.rcParams["figure.dpi"] = 110


def plot_signal_vs_ttf(
    signal: np.ndarray, time_to_failure: np.ndarray, out_path: Path, n_points: int | None = 150_000
) -> Path:
    """Grafica la senal acustica cruda y ``time_to_failure`` en dos paneles con eje x compartido."""
    if n_points is not None:
        signal = signal[:n_points]
        time_to_failure = time_to_failure[:n_points]

    fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
    axes[0].plot(signal, color="#1f77b4", linewidth=0.4)
    axes[0].set_ylabel("acoustic_data")
    axes[0].set_title("Senal acustica cruda vs. time_to_failure")

    axes[1].plot(time_to_failure, color="#d62728", linewidth=1.2)
    axes[1].set_ylabel("time_to_failure")
    axes[1].set_xlabel("Indice de muestra")

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def plot_ttf_distribution(time_to_failure: np.ndarray, out_path: Path) -> Path:
    """Grafica el histograma/densidad de ``time_to_failure``."""
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(time_to_failure, bins=40, color="#2ca02c", alpha=0.75, density=True, edgecolor="white")
    ax.set_xlabel("time_to_failure")
    ax.set_ylabel("Densidad")
    ax.set_title("Distribucion de time_to_failure")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def plot_spectrogram(signal: np.ndarray, out_path: Path, nperseg: int = 2048) -> Path:
    """Grafica el espectrograma de un segmento de senal de ejemplo."""
    freqs, times, sxx = spectrogram(signal.astype(np.float64), nperseg=nperseg)
    fig, ax = plt.subplots(figsize=(9, 5))
    pcm = ax.pcolormesh(times, freqs, 10 * np.log10(sxx + 1e-12), shading="gouraud", cmap="magma")
    ax.set_xlabel("Tiempo (muestras del segmento)")
    ax.set_ylabel("Frecuencia (normalizada)")
    ax.set_title("Espectrograma de un segmento de ejemplo")
    fig.colorbar(pcm, ax=ax, label="Potencia (dB)")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def plot_mae_comparison(mae_by_model: dict[str, float], out_path: Path) -> Path:
    """Grafica un barchart comparando el MAE de validacion de cada modelo."""
    items = sorted(mae_by_model.items(), key=lambda kv: kv[1])
    names = [k for k, _ in items]
    values = [v for _, v in items]
    colors = ["#9467bd" if n == "ensemble" else "#1f77b4" for n in names]

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(names, values, color=colors)
    ax.set_ylabel("MAE (out-of-fold)")
    ax.set_title("Comparacion de MAE por modelo")
    ax.tick_params(axis="x", rotation=30)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value, f"{value:.3f}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def plot_prediction_vs_actual(
    y_true: np.ndarray, predictions_by_model: dict[str, np.ndarray], out_path: Path
) -> Path:
    """Grafica un scatter de prediccion vs. valor real para uno o mas modelos."""
    n_models = len(predictions_by_model)
    fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 5.5), squeeze=False)
    axes = axes[0]

    lims = [float(np.min(y_true)), float(np.max(y_true))]
    for ax, (name, preds) in zip(axes, predictions_by_model.items()):
        ax.scatter(y_true, preds, alpha=0.6, s=18, color="#1f77b4", edgecolor="none")
        ax.plot(lims, lims, color="#d62728", linestyle="--", linewidth=1.0, label="y = x")
        ax.set_xlabel("time_to_failure real")
        ax.set_ylabel("time_to_failure predicho")
        ax.set_title(f"Prediccion vs. real ({name})")
        ax.legend()

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def plot_feature_importance(
    feature_names: list[str], importances: np.ndarray, out_path: Path, top_n: int = 20, title: str = "Feature importance"
) -> Path:
    """Grafica las ``top_n`` features mas importantes de un modelo entrenado."""
    order = np.argsort(importances)[::-1][:top_n]
    top_names = [feature_names[i] for i in order][::-1]
    top_values = importances[order][::-1]

    fig, ax = plt.subplots(figsize=(8, max(4, 0.3 * top_n)))
    ax.barh(top_names, top_values, color="#ff7f0e")
    ax.set_xlabel("Importancia")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def plot_activation_comparison(mae_by_activation: dict[str, float], out_path: Path) -> Path:
    """Grafica un barchart comparando el MAE de validacion de la CNN 1D segun activacion."""
    items = sorted(mae_by_activation.items(), key=lambda kv: kv[1])
    names = [k for k, _ in items]
    values = [v for _, v in items]

    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(names, values, color="#e377c2")
    ax.set_ylabel("MAE (validacion)")
    ax.set_title("CNN 1D: comparacion de activaciones (ReLU vs. GELU vs. Swish)")
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value, f"{value:.3f}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def plot_activation_loss_curves(epoch_losses_by_activation: dict[str, list[float]], out_path: Path) -> Path:
    """Grafica la evolucion de la loss (Huber ponderada) por epoca para cada activacion."""
    fig, ax = plt.subplots(figsize=(8, 5))
    for name, losses in epoch_losses_by_activation.items():
        epochs = np.arange(1, len(losses) + 1)
        ax.plot(epochs, losses, marker="o", label=name)

    ax.set_xlabel("Epoca")
    ax.set_ylabel("Weighted Huber loss (entrenamiento)")
    ax.set_title("CNN 1D: loss por epoca segun activacion")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def plot_cv_mae_by_fold(fold_scores_by_model: dict[str, list[float]], out_path: Path) -> Path:
    """Grafica la evolucion del MAE por fold de cross-validation para cada modelo."""
    fig, ax = plt.subplots(figsize=(8, 5))
    for name, scores in fold_scores_by_model.items():
        folds = np.arange(1, len(scores) + 1)
        ax.plot(folds, scores, marker="o", label=name)

    ax.set_xlabel("Fold")
    ax.set_ylabel("MAE de validacion")
    ax.set_title("MAE por fold de cross-validation (GroupKFold)")
    ax.set_xticks(np.arange(1, max(len(s) for s in fold_scores_by_model.values()) + 1))
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def _subsample_indices(n_total: int, n_frames: int) -> np.ndarray:
    """Indices (crecientes, sin repetir) para muestrear ``n_total`` puntos reales en ``n_frames``."""
    n_frames = min(n_frames, n_total)
    return np.unique(np.linspace(0, n_total - 1, n_frames).astype(int))


def animate_signal_vs_ttf(
    signal: np.ndarray, time_to_failure: np.ndarray, out_path: Path, n_points: int | None = 150_000, n_frames: int = 60
) -> Path:
    """Genera un GIF tipo 'racing line chart' de la senal acustica y time_to_failure reales.

    Usa los mismos datos ya calculados que ``plot_signal_vs_ttf`` (submuestreados a
    ``n_frames`` cuadros), sin fabricar valores nuevos.
    """
    if n_points is not None:
        signal = signal[:n_points]
        time_to_failure = time_to_failure[:n_points]

    x = np.arange(len(signal))
    frame_idx = _subsample_indices(len(signal), n_frames)

    with plt.style.context("dark_background"):
        fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

        line0, = axes[0].plot([], [], color="#4fc3f7", linewidth=0.6)
        label0 = axes[0].annotate(
            "", xy=(0, 0), xytext=(15, 15), textcoords="offset points",
            color="black", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", fc="#4fc3f7", ec="none"),
        )
        axes[0].set_xlim(0, x[-1])
        axes[0].set_ylim(float(np.min(signal)), float(np.max(signal)))
        axes[0].set_ylabel("acoustic_data")
        axes[0].set_title("Senal acustica cruda vs. time_to_failure (animado)")

        line1, = axes[1].plot([], [], color="#ff8a65", linewidth=1.4)
        label1 = axes[1].annotate(
            "", xy=(0, 0), xytext=(15, -20), textcoords="offset points",
            color="black", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", fc="#ff8a65", ec="none"),
        )
        axes[1].set_xlim(0, x[-1])
        axes[1].set_ylim(float(np.min(time_to_failure)), float(np.max(time_to_failure)))
        axes[1].set_ylabel("time_to_failure")
        axes[1].set_xlabel("Indice de muestra")

        fig.tight_layout()

        def update(frame: int):
            i = frame_idx[frame]
            line0.set_data(x[: i + 1], signal[: i + 1])
            label0.xy = (x[i], signal[i])
            label0.set_text(f"acoustic_data = {signal[i]:.1f}")

            line1.set_data(x[: i + 1], time_to_failure[: i + 1])
            label1.xy = (x[i], time_to_failure[i])
            label1.set_text(f"time_to_failure = {time_to_failure[i]:.3f}")
            return line0, label0, line1, label1

        ani = FuncAnimation(fig, update, frames=len(frame_idx), interval=120, blit=False)
        ani.save(out_path, writer="pillow")
        plt.close(fig)

    return out_path


def animate_activation_loss_curves(
    epoch_losses_by_activation: dict[str, list[float]], out_path: Path, n_frames: int = 40
) -> Path:
    """Genera un GIF tipo 'racing line chart' de la loss por epoca de la CNN 1D, por activacion.

    Usa los mismos valores reales de ``epoch_losses_by_activation`` que
    ``plot_activation_loss_curves``; el numero de cuadros se limita al numero de
    epocas cuando hay menos epocas que ``n_frames``.
    """
    max_epochs = max(len(losses) for losses in epoch_losses_by_activation.values())
    n_frames = min(n_frames, max_epochs)
    all_losses = np.concatenate([np.asarray(v, dtype=float) for v in epoch_losses_by_activation.values()])

    colors = ["#4fc3f7", "#ff8a65", "#aed581", "#ba68c8", "#ffd54f"]

    with plt.style.context("dark_background"):
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.set_xlim(1, max_epochs)
        ax.set_ylim(float(all_losses.min()) * 0.95, float(all_losses.max()) * 1.05)
        ax.set_xlabel("Epoca")
        ax.set_ylabel("Weighted Huber loss (entrenamiento)")
        ax.set_title("CNN 1D: loss por epoca segun activacion (animado)")

        lines = {}
        labels = {}
        for i, name in enumerate(epoch_losses_by_activation):
            color = colors[i % len(colors)]
            (lines[name],) = ax.plot([], [], marker="o", color=color, label=name)
            labels[name] = ax.annotate(
                "", xy=(1, 0), xytext=(10, 10), textcoords="offset points",
                color="black", fontsize=9,
                bbox=dict(boxstyle="round,pad=0.3", fc=color, ec="none"),
            )
        ax.legend(loc="upper right")
        fig.tight_layout()

        def update(frame: int):
            epoch_count = frame + 1
            artists = []
            for name, losses in epoch_losses_by_activation.items():
                k = min(epoch_count, len(losses))
                epochs = np.arange(1, k + 1)
                values = losses[:k]
                lines[name].set_data(epochs, values)
                labels[name].xy = (epochs[-1], values[-1])
                labels[name].set_text(f"{name}: {values[-1]:.3f}")
                artists.extend([lines[name], labels[name]])
            return artists

        ani = FuncAnimation(fig, update, frames=n_frames, interval=200, blit=False)
        ani.save(out_path, writer="pillow")
        plt.close(fig)

    return out_path
