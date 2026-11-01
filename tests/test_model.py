"""Unit tests for model pipeline encapsulation, inference, and calibration."""

import pytest

from src.model import CirrhosisPredictionPipeline, determine_risk_level
from src.preprocessing import DEFAULT_FEATURE_NAMES


def test_model_loading_and_attributes(model_pipeline: CirrhosisPredictionPipeline):
    """Verify that the model pipeline is loaded with required sub-components."""
    assert model_pipeline is not None
    assert hasattr(model_pipeline, "preprocessor")
    assert hasattr(model_pipeline, "classifier")
    assert hasattr(model_pipeline, "decision_threshold")
    assert 0.0 < model_pipeline.decision_threshold < 1.0
    assert model_pipeline.feature_names == DEFAULT_FEATURE_NAMES


def test_missing_model_file_raises():
    """Verify loading from a missing file raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        CirrhosisPredictionPipeline.load("nonexistent/path/model.joblib")


def test_predict_proba_bounds(
    model_pipeline: CirrhosisPredictionPipeline,
    synthetic_healthy_patient: dict,
    synthetic_high_risk_patient: dict,
):
    """Ensure predicted probabilities strictly fall between 0.0 and 1.0."""
    probs = model_pipeline.predict_proba([synthetic_healthy_patient, synthetic_high_risk_patient])
    assert len(probs) == 2
    for p in probs:
        assert 0.0 <= p <= 1.0


def test_threshold_decision_logic(
    model_pipeline: CirrhosisPredictionPipeline,
    synthetic_healthy_patient: dict,
    synthetic_high_risk_patient: dict,
):
    """Ensure binary decisions obey the configured decision threshold."""
    probs = model_pipeline.predict_proba([synthetic_healthy_patient, synthetic_high_risk_patient])
    preds = model_pipeline.predict([synthetic_healthy_patient, synthetic_high_risk_patient])

    for prob, pred in zip(probs, preds):
        expected_pred = int(prob >= model_pipeline.decision_threshold)
        assert pred == expected_pred


def test_predict_detailed_keys(
    model_pipeline: CirrhosisPredictionPipeline,
    synthetic_high_risk_patient: dict,
):
    """Ensure detailed predictions return all required fields and valid types."""
    details = model_pipeline.predict_detailed([synthetic_high_risk_patient])
    assert len(details) == 1
    res = details[0]

    assert "prediction" in res
    assert "prediction_label" in res
    assert "cirrhosis_probability" in res
    assert "decision_threshold" in res
    assert "risk_level" in res
    assert "model_version" in res

    assert res["prediction"] in (0, 1)
    assert res["prediction_label"] in ("High Risk", "Low Risk")
    assert res["risk_level"] in ("Low", "Moderate", "High", "Critical")


def test_determine_risk_level_boundaries():
    """Verify probability risk stratification function."""
    assert determine_risk_level(0.10) == "Low"
    assert determine_risk_level(0.29) == "Low"
    assert determine_risk_level(0.35) == "Moderate"
    assert determine_risk_level(0.59) == "Moderate"
    assert determine_risk_level(0.65) == "High"
    assert determine_risk_level(0.84) == "High"
    assert determine_risk_level(0.95) == "Critical"


def test_model_info_content(model_pipeline: CirrhosisPredictionPipeline):
    """Verify get_info() produces expected metadata payload."""
    info = model_pipeline.get_info()
    assert info["model_name"] == "Cirrhosis-CatBoost-Calibrated"
    assert "version" in info
    assert "feature_names" in info
    assert "decision_threshold" in info
    assert "training_metrics" in info
