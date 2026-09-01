"""Genera versiones animadas (GIF) de los graficos de linea/serie temporal
del notebook 01_Vibration_EDA_and_RUL.ipynb, usando exactamente los mismos
datos reales (mismo experimento, mismos archivos, mismo procesamiento) que
generan los PNG estaticos. No se inventan valores: se resamplean los mismos
arreglos ya calculados a ~30-60 frames.

    python -m src.generate_animations
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

from src.ingest import EXPERIMENTS

ROOT = Path(__file__).resolve().parent.parent
REPORTS_DIR = ROOT / "outputs" / "reports"

N_FRAMES = 45


def _subsample_frame_indices(n_points: int, n_frames: int) -> np.ndarray:
    """Indices crecientes (reveal progresivo) que llegan hasta n_points."""
    n_frames = min(n_frames, n_points)
    return np.unique(np.linspace(3, n_points, n_frames, dtype=int))


def animate_rms_degradation_curve(out_path: Path) -> None:
    """RMS de vibracion a lo largo del experimento 2 (misma logica que la
    celda 7 del notebook): serie real que crece hasta la falla."""
    folder = EXPERIMENTS["2nd_test"]
    files = sorted(folder.iterdir(), key=lambda p: p.name)

    rms_trend = []
    for f in files[::10]:
        sig = np.loadtxt(f)[:, 0]
        rms_trend.append(np.sqrt(np.mean(sig**2)))
    rms_trend = np.array(rms_trend)
    x = np.arange(len(rms_trend))

    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(12, 6))
    line, = ax.plot([], [], color="#DD8452", linewidth=2)
    ax.axvline(len(rms_trend) - 1, color="#C44E52", linestyle="--", alpha=0.6, label="Falla")
    ax.set_xlim(0, len(rms_trend) - 1)
    ax.set_ylim(rms_trend.min() * 0.9, rms_trend.max() * 1.15)
    ax.set_title("RMS de vibracion a lo largo del experimento 2 (cada 10mo snapshot)")
    ax.set_xlabel("Snapshot (orden temporal)")
    ax.set_ylabel("RMS")
    ax.legend(loc="upper left")

    label = ax.annotate(
        "", xy=(0, 0), xytext=(15, 15), textcoords="offset points",
        bbox=dict(boxstyle="round,pad=0.4", fc="#DD8452", ec="white", alpha=0.9),
        color="black", fontsize=10, fontweight="bold",
    )

    frame_idx = _subsample_frame_indices(len(rms_trend), N_FRAMES)

    def update(i):
        n = frame_idx[i]
        line.set_data(x[:n], rms_trend[:n])
        cur_x, cur_y = x[n - 1], rms_trend[n - 1]
        label.set_position((15, 15))
        label.xy = (cur_x, cur_y)
        label.set_text(f"RMS: {cur_y:.3f}")
        return line, label

    ani = FuncAnimation(fig, update, frames=len(frame_idx), interval=120, blit=False)
    ani.save(out_path, writer="pillow")
    plt.close(fig)
    plt.style.use("default")


def animate_raw_signal_early_vs_late(out_path: Path) -> None:
    """Senal cruda: inicio (sano) vs justo antes de la falla (degradado),
    misma logica que la celda 3 del notebook."""
    folder = EXPERIMENTS["2nd_test"]
    files = sorted(folder.iterdir(), key=lambda p: p.name)
    early_file, late_file = files[0], files[-1]

    early_signal = np.loadtxt(early_file)[:, 0][:2000]
    late_signal = np.loadtxt(late_file)[:, 0][:2000]
    x = np.arange(len(early_signal))

    plt.style.use("dark_background")
    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharey=True)

    line_early, = axes[0].plot([], [], color="#4C72B0", linewidth=0.7)
    line_late, = axes[1].plot([], [], color="#C44E52", linewidth=0.7)
    axes[0].set_title(f"Inicio del experimento ({early_file.name}) - rodamiento sano")
    axes[1].set_title(f"Justo antes de la falla ({late_file.name}) - rodamiento degradado")
    axes[1].set_xlabel("Muestra (primeros 2000 puntos, 20kHz)")
    for ax in axes:
        ax.set_xlim(0, len(x) - 1)
    ymin = min(early_signal.min(), late_signal.min()) * 1.1
    ymax = max(early_signal.max(), late_signal.max()) * 1.1
    axes[0].set_ylim(ymin, ymax)

    label_early = axes[0].annotate(
        "", xy=(0, 0), xytext=(15, 15), textcoords="offset points",
        bbox=dict(boxstyle="round,pad=0.4", fc="#4C72B0", ec="white", alpha=0.9),
        color="white", fontsize=9, fontweight="bold",
    )
    label_late = axes[1].annotate(
        "", xy=(0, 0), xytext=(15, 15), textcoords="offset points",
        bbox=dict(boxstyle="round,pad=0.4", fc="#C44E52", ec="white", alpha=0.9),
        color="white", fontsize=9, fontweight="bold",
    )

    frame_idx = _subsample_frame_indices(len(x), N_FRAMES)

    def update(i):
        n = frame_idx[i]
        line_early.set_data(x[:n], early_signal[:n])
        line_late.set_data(x[:n], late_signal[:n])
        label_early.xy = (x[n - 1], early_signal[n - 1])
        label_early.set_text(f"sano: {early_signal[n - 1]:.3f}")
        label_late.xy = (x[n - 1], late_signal[n - 1])
        label_late.set_text(f"degradado: {late_signal[n - 1]:.3f}")
        return line_early, line_late, label_early, label_late

    ani = FuncAnimation(fig, update, frames=len(frame_idx), interval=120, blit=False)
    ani.save(out_path, writer="pillow")
    plt.close(fig)
    plt.style.use("default")


def main() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    print("[1/2] Generando rms_degradation_curve_animated.gif...")
    animate_rms_degradation_curve(REPORTS_DIR / "rms_degradation_curve_animated.gif")
    print("[2/2] Generando raw_signal_early_vs_late_animated.gif...")
    animate_raw_signal_early_vs_late(REPORTS_DIR / "raw_signal_early_vs_late_animated.gif")
    print(f"Guardado en: {REPORTS_DIR}")


if __name__ == "__main__":
    main()
