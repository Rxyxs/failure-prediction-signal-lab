"""Ventaneo de la senal acustica en bloques de tamano fijo.

Divide la senal cruda (y su correspondiente ``time_to_failure``) en bloques
no solapados de ``segment_size`` muestras, replicando la estructura de
segmentos del dataset original de LANL Earthquake Prediction.

Autor: Pablo Reyes
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def create_windows(df: pd.DataFrame, segment_size: int) -> list[dict]:
    """Divide ``df`` en ventanas no solapadas de tamano ``segment_size``.

    Cada ventana conserva la senal cruda completa (``acoustic_data``) y el
    valor de ``time_to_failure`` tomado al final del bloque, que es el
    objetivo a predecir tal como en la competencia original.

    Retorna una lista de diccionarios con llaves ``segment_id``, ``signal`` y
    ``time_to_failure``.
    """
    n_points = len(df)
    n_windows = n_points // segment_size

    acoustic = df["acoustic_data"].to_numpy()
    ttf = df["time_to_failure"].to_numpy()

    windows = []
    for i in range(n_windows):
        start = i * segment_size
        end = start + segment_size
        windows.append(
            {
                "segment_id": i,
                "signal": acoustic[start:end].astype(np.float32),
                "time_to_failure": float(ttf[end - 1]),
            }
        )
    return windows
