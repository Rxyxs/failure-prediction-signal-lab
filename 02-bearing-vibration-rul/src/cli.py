"""CLI de inferencia: dado un archivo de snapshot de vibracion (formato IMS,
20.480 puntos), predice minutos restantes hasta falla.

    python -m src.cli --file path/al/snapshot.txt
"""

from __future__ import annotations

import argparse

import joblib
import numpy as np
import pandas as pd

from src.features import extract_features

MODEL_PATH = "outputs/models/best_model.joblib"
FEATURE_ORDER_PATH = "outputs/models/feature_order.joblib"


def predict_file(path: str) -> float:
    arr = np.loadtxt(path)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    signal = arr[:, np.argmax(arr.std(axis=0))]
    feats = extract_features(signal)

    model = joblib.load(MODEL_PATH)
    feature_order = joblib.load(FEATURE_ORDER_PATH)
    X = pd.DataFrame([feats])[feature_order]
    return float(model.predict(X)[0])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", required=True, help="Archivo de snapshot de vibracion")
    args = parser.parse_args()

    minutes = predict_file(args.file)
    print(f"Tiempo restante estimado hasta falla: {minutes:.1f} minutos ({minutes/60:.2f} horas)")


if __name__ == "__main__":
    main()
