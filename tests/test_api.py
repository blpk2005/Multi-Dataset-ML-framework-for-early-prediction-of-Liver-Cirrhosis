"""Integration and end-to-end tests for FastAPI endpoints."""

from starlette.testclient import TestClient


def test_root_endpoint(client: TestClient):
    """Verify root index endpoint responds with 200 and directory links."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "name" in data
    assert "version" in data
    assert "health" in data
    assert "predict" in data["documentation"] or "docs" in data["documentation"]


def test_health_check(client: TestClient):
    """Verify /health endpoint returns service health status."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "version" in data
    assert "environment" in data


def test_readiness_check(client: TestClient):
    """Verify /ready endpoint confirms ML pipeline is loaded."""
    response = client.get("/ready")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ready"
    assert data["model_loaded"] is True
    assert isinstance(data["feature_names"], list)
    assert data["decision_threshold"] > 0


def test_model_info_endpoint(client: TestClient):
    """Verify /model-info delivers model provenance and research disclaimer."""
    response = client.get("/model-info")
    assert response.status_code == 200
    data = response.json()
    assert "model_name" in data
    assert "version" in data
    assert "feature_names" in data
    assert "disclaimer" in data
    assert "RESEARCH NOTICE" in data["disclaimer"]


def test_predict_valid_synthetic_input(client: TestClient, synthetic_healthy_patient: dict):
    """Verify /predict processes a valid clinical record correctly."""
    response = client.post("/predict", json=synthetic_healthy_patient)
    assert response.status_code == 200
    data = response.json()

    assert data["prediction"] in (0, 1)
    assert data["prediction_label"] in ("High Risk", "Low Risk")
    assert 0.0 <= data["cirrhosis_probability"] <= 1.0
    assert data["risk_level"] in ("Low", "Moderate", "High", "Critical")
    assert "disclaimer" in data
    assert "RESEARCH NOTICE" in data["disclaimer"]


def test_predict_with_missing_clinical_values(
    client: TestClient, synthetic_patient_with_missing_values: dict
):
    """Verify /predict handles omitted/null laboratory values via median imputation."""
    response = client.post("/predict", json=synthetic_patient_with_missing_values)
    assert response.status_code == 200
    data = response.json()
    assert data["prediction"] in (0, 1)
    assert 0.0 <= data["cirrhosis_probability"] <= 1.0


def test_predict_invalid_inputs(client: TestClient, synthetic_invalid_patients: list):
    """Verify /predict rejects physiological out-of-bounds or malformed inputs with 422."""
    for case in synthetic_invalid_patients:
        response = client.post("/predict", json=case["payload"])
        assert response.status_code == 422, f"Failed on case: {case['desc']}"
        data = response.json()
        assert "error" in data
        assert "details" in data
        assert len(data["details"]) > 0


def test_predict_batch(
    client: TestClient,
    synthetic_healthy_patient: dict,
    synthetic_high_risk_patient: dict,
):
    """Verify /predict/batch processes multiple records in one request."""
    batch_payload = {
        "patients": [synthetic_healthy_patient, synthetic_high_risk_patient]
    }
    response = client.post("/predict/batch", json=batch_payload)
    assert response.status_code == 200
    data = response.json()
    assert data["count"] == 2
    assert len(data["predictions"]) == 2
    assert "disclaimer" in data


def test_prometheus_metrics_endpoint(client: TestClient, synthetic_healthy_patient: dict):
    """Verify Prometheus scrape endpoint publishes telemetry."""
    # Generate at least one request to increment counters
    client.post("/predict", json=synthetic_healthy_patient)

    response = client.get("/metrics")
    assert response.status_code == 200
    metrics_text = response.text
    assert "cirrhosis_predictions_total" in metrics_text
    assert "cirrhosis_model_loaded" in metrics_text
