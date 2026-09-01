"""Ingesta de senal cruda: NASA IMS Bearing Dataset (Center for Intelligent
Maintenance Systems, Univ. of Cincinnati) -- 3 experimentos reales de
run-to-failure sobre rodamientos, 20kHz, snapshots de 20.480 puntos cada
~10 min hasta la falla fisica del rodamiento.

Cada experimento (1st_test/2nd_test/4th_test) es un grupo independiente
para GroupKFold: no hay fuga posible entre experimentos porque son
corridas fisicas distintas, en fechas distintas, con rodamientos distintos.

Para cada snapshot se usa el canal de MAYOR desviacion estandar (proxy
automatico y explicable de "el rodamiento mas degradado en este instante",
sin necesitar hardcodear cual canal es el que efectivamente falla en cada
experimento -- una decision de diseno documentada, no una simplificacion
oculta).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

DATA_ROOT = Path(r"C:\Users\preye\.cache\kagglehub\datasets\vinayak123tyagi\bearing-dataset\versions\1")

EXPERIMENTS = {
    "1st_test": DATA_ROOT / "1st_test" / "1st_test",
    "2nd_test": DATA_ROOT / "2nd_test" / "2nd_test",
    "4th_test": DATA_ROOT / "3rd_test" / "4th_test" / "txt",
}

SNAPSHOT_INTERVAL_MINUTES = 10
SAMPLING_RATE_HZ = 20_000


@dataclass
class Snapshot:
    experiment: str
    file_index: int
    n_files_in_experiment: int
    signal: np.ndarray  # canal de mayor std en este snapshot


def iter_snapshots(experiment: str, stride: int = 1) -> list[Snapshot]:
    """Lee cada archivo del experimento (cada `stride`-esimo, para acotar
    tiempo de computo) y extrae el canal de mayor varianza."""
    folder = EXPERIMENTS[experiment]
    files = sorted(folder.iterdir(), key=lambda p: p.name)
    n_total = len(files)

    snapshots = []
    for i, path in enumerate(files):
        if i % stride != 0:
            continue
        arr = np.loadtxt(path)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        worst_channel = arr[:, np.argmax(arr.std(axis=0))]
        snapshots.append(Snapshot(experiment, i, n_total, worst_channel))
    return snapshots


def time_to_failure_minutes(snap: Snapshot) -> float:
    """El ultimo snapshot de cada experimento = momento de falla fisica
    (parada del banco de pruebas por vibracion excesiva, documentado por
    IMS/Univ. Cincinnati). RUL = archivos restantes * intervalo entre snapshots."""
    remaining_files = snap.n_files_in_experiment - 1 - snap.file_index
    return remaining_files * SNAPSHOT_INTERVAL_MINUTES


def time_to_failure_fraction(snap: Snapshot) -> float:
    """RUL normalizada [0,1]: fraccion de vida restante, invariante a la
    duracion absoluta del experimento. Necesaria porque los 3 experimentos
    duran radicalmente distinto (1st_test ~15 dias, 2nd_test ~7 dias,
    4th_test ~44 dias) -- un modelo entrenado en minutos absolutos sobre 2
    experimentos nunca aprende la escala temporal del tercero, y falla en
    GroupKFold (ver Nota de depuracion en el README)."""
    remaining_files = snap.n_files_in_experiment - 1 - snap.file_index
    return remaining_files / (snap.n_files_in_experiment - 1)


def load_all_snapshots(stride: int = 3) -> list[Snapshot]:
    all_snaps = []
    for exp in EXPERIMENTS:
        all_snaps.extend(iter_snapshots(exp, stride=stride))
    return all_snaps
