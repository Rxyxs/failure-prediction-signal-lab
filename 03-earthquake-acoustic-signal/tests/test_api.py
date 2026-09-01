"""Pruebas del endpoint /predict y /health de la API FastAPI.

Autor: Pablo Reyes
"""
from __future__ import annotations

import numpy as np
import pytest
from fastapi.testclient import TestClient

from src.api.app import app

client = TestClient(app)


def test_health_endpoint_returns_status():
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert "status" in body
    assert body["status"] == "ok"


def test_predict_endpoint_with_valid_signal():
    signal = list(np.random.default_rng(0).normal(0, 2.0, 5000))
    response = client.post("/predict", json={"signal": signal})

    # Si no hay modelo entrenado en el entorno de pruebas, se espera 503; si lo
    # hay (por ejemplo tras correr el pipeline), se espera 200 con una prediccion valida.
    assert response.status_code in (200, 503)
    if response.status_code == 200:
        body = response.json()
        assert "time_to_failure" in body
        assert isinstance(body["time_to_failure"], float)
        assert body["n_samples"] == len(signal)


def test_predict_endpoint_rejects_empty_signal():
    response = client.post("/predict", json={"signal": []})
    assert response.status_code == 400


def test_predict_endpoint_rejects_malformed_payload():
    response = client.post("/predict", json={"not_signal": [1, 2, 3]})
    assert response.status_code == 422
