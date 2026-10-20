import pytest
from fastapi.testclient import TestClient

import main as api_module

VALID_PATIENT = {
    "age": 63, "sex": 1, "cp": 1, "trestbps": 145, "chol": 233,
    "fbs": 1, "restecg": 2, "thalach": 150, "exang": 0,
    "oldpeak": 2.3, "slope": 3, "ca": 0, "thal": 6,
}


@pytest.fixture
def client(trained_model):
    api_module.model_state["model"] = trained_model
    yield TestClient(api_module.app)
    api_module.model_state.clear()


def test_health_reports_model_loaded(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "model_loaded": True}


def test_predict_returns_prediction_and_confidence(client):
    response = client.post("/predict", json=VALID_PATIENT)
    assert response.status_code == 200
    body = response.json()
    assert body["prediction"] in (0, 1)
    assert 0.5 <= body["confidence"] <= 1.0
    assert body["label"] in ("disease", "no disease")


def test_predict_rejects_missing_required_field(client):
    incomplete = {k: v for k, v in VALID_PATIENT.items() if k != "age"}
    assert client.post("/predict", json=incomplete).status_code == 422


def test_predict_rejects_out_of_range_value(client):
    assert client.post("/predict", json={**VALID_PATIENT, "trestbps": 9999}).status_code == 422


def test_predict_rejects_invalid_thal_literal(client):
    assert client.post("/predict", json={**VALID_PATIENT, "thal": 4}).status_code == 422


def test_metrics_endpoint_exposes_prometheus_counters(client):
    client.post("/predict", json=VALID_PATIENT)
    body = client.get("/metrics").text
    assert "heart_api_http_requests_total" in body
    assert "heart_api_predictions_total" in body


def test_response_includes_request_id_header(client):
    response = client.get("/health")
    assert "X-Request-ID" in response.headers
