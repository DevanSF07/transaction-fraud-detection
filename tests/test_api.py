"""Tests for FastAPI HTTP endpoints."""

import pytest
from fastapi.testclient import TestClient
from fraud_detection.serving.api import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["version"] == "1.0.0"
    assert data["pipeline_fitted"] is True


def test_predict_single_endpoint(client):
    payload = {
        "transaction_id": "tx_test_999",
        "Time": 3600.0,
        "Amount": 149.50,
        "V1": -1.35, "V2": -0.07, "V3": 2.53, "V4": 1.37,
        "V5": -0.33, "V6": 0.46, "V7": 0.23, "V8": 0.09,
        "V9": 0.36, "V10": 0.09, "V11": -0.55, "V12": -0.61,
        "V13": -0.99, "V14": -0.31, "V15": 1.46, "V16": -0.47,
        "V17": 0.20, "V18": 0.02, "V19": 0.40, "V20": 0.25,
        "V21": 0.21, "V22": 0.77, "V23": 0.90, "V24": -0.68,
        "V25": -0.32, "V26": -0.13, "V27": -0.02, "V28": 0.01
    }
    response = client.post("/v1/predict", json=payload)
    assert response.status_code == 200
    res = response.json()
    assert "fraud_probability" in res
    assert "risk_level" in res
    assert res["transaction_id"] == "tx_test_999"


def test_explain_endpoint(client):
    payload = {
        "transaction_id": "tx_test_explain",
        "Time": 7200.0,
        "Amount": 890.00,
        "V4": 3.8, "V11": 3.2, "V12": -4.1, "V14": -4.8, "V17": -4.5
    }
    for i in range(1, 29):
        key = f"V{i}"
        if key not in payload:
            payload[key] = 0.0

    response = client.post("/v1/explain", json=payload)
    assert response.status_code == 200
    res = response.json()
    assert "top_shap_factors" in res
    assert "lime_decision_rules" in res
    assert "analyst_summary" in res
