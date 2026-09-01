"""Genera las versiones animadas (GIF) de los graficos de series temporales reales.

Reutiliza exactamente los mismos datos ya calculados que ``scripts/generate_figures.py``
y ``scripts/run_activation_experiment.py`` (senal sintetica con semilla fija, y el
reporte ``activation_experiment.json`` ya persistido), sin fabricar valores nuevos.

Genera:
- reports/figures/signal_vs_ttf_animated.gif
- reports/figures/activation_loss_curves_animated.gif

Ejecutar con: ``python -m scripts.generate_animations``

Autor: Pablo Reyes
"""
from __future__ import annotations

import json

from src.config import REPORTS_DIR, SyntheticConfig, ensure_directories
from src.data.synthetic import generate_synthetic_dataset
from src.viz.plots import animate_activation_loss_curves, animate_signal_vs_ttf

FIGURES_DIR = REPORTS_DIR / "figures"


def main() -> None:
    ensure_directories()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    print("Generando signal_vs_ttf_animated.gif ...")
    raw_df = generate_synthetic_dataset(SyntheticConfig(n_segments=6, segment_size=150_000))
    signal = raw_df["acoustic_data"].to_numpy()
    ttf = raw_df["time_to_failure"].to_numpy()
    animate_signal_vs_ttf(signal, ttf, FIGURES_DIR / "signal_vs_ttf_animated.gif", n_points=300_000, n_frames=60)

    activation_report_path = REPORTS_DIR / "activation_experiment.json"
    if activation_report_path.exists():
        print("Generando activation_loss_curves_animated.gif ...")
        with open(activation_report_path, "r", encoding="utf-8") as fh:
            report = json.load(fh)
        animate_activation_loss_curves(
            report["epoch_losses_by_activation"], FIGURES_DIR / "activation_loss_curves_animated.gif"
        )
    else:
        print(f"Aviso: {activation_report_path} no existe; se omite activation_loss_curves_animated.gif")

    print(f"\nGIFs guardados en {FIGURES_DIR}")


if __name__ == "__main__":
    main()
