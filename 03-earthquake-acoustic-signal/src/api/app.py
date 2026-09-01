"""API FastAPI para prediccion de ``time_to_failure`` a partir de una senal acustica cruda.

Expone ``POST /predict`` (recibe un chunk de floats y devuelve la estimacion
de tiempo hasta la falla) y ``GET /health`` para verificacion de estado del
servicio. El modelo utilizado es el mejor modelo tabular entrenado por el
pipeline (LightGBM por defecto, con fallback a Ridge si no esta disponible).

Autor: Pablo Reyes
"""
from __future__ import annotations

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.config import FeatureConfig, MODELS_DIR
from src.features.build_features import extract_segment_features

app = FastAPI(
    title="LANL Earthquake Signal Prediction API",
    description="Servicio de prediccion de time_to_failure a partir de senales acusticas.",
    version="1.0.0",
)

_MODEL_CANDIDATES = ["lightgbm", "catboost", "random_forest", "ridge", "lasso"]


class PredictRequest(BaseModel):
    """Payload de entrada para el endpoint de prediccion."""

    signal: list[float] = Field(..., description="Chunk de senal acustica cruda (lista de floats).")


class PredictResponse(BaseModel):
    """Respuesta del endpoint de prediccion."""

    time_to_failure: float
    model_used: str
    n_samples: int


def _load_available_model():
    """Carga el primer modelo disponible en ``MODELS_DIR`` segun orden de preferencia."""
    for name in _MODEL_CANDIDATES:
        path = MODELS_DIR / f"{name}.joblib"
        if path.exists():
            feature_cols_path = MODELS_DIR / "feature_columns.joblib"
            if not feature_cols_path.exists():
                continue
            model = joblib.load(path)
            feature_cols = joblib.load(feature_cols_path)
            return name, model, feature_cols
    return None, None, None


@app.get("/health")
def health() -> dict:
    """Endpoint de verificacion de estado del servicio."""
    model_name, model, _ = _load_available_model()
    return {
        "status": "ok",
        "model_available": model is not None,
        "model_name": model_name,
    }


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest) -> PredictResponse:
    """Predice ``time_to_failure`` a partir de un chunk de senal acustica cruda."""
    if not request.signal:
        raise HTTPException(status_code=400, detail="La senal no puede estar vacia.")

    model_name, model, feature_cols = _load_available_model()
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="No hay ningun modelo entrenado disponible. Ejecute el pipeline de entrenamiento primero.",
        )

    signal = np.asarray(request.signal, dtype=np.float64)
    features = extract_segment_features(signal, FeatureConfig())
    feature_vector = np.array([[features.get(col, 0.0) for col in feature_cols]])

    prediction = float(model.predict(feature_vector)[0])

    return PredictResponse(
        time_to_failure=prediction,
        model_used=model_name,
        n_samples=len(request.signal),
    )
