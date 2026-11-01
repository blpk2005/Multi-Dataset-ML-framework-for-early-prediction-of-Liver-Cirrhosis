"""Pytest configuration, fixtures, and synthetic test datasets."""

import sys
from pathlib import Path
from typing import Dict, Any, Generator
import pytest
from starlette.testclient import TestClient

# Ensure root directory is on PYTHONPATH
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from api.main import create_app  # noqa: E402
from src.config import settings  # noqa: E402
from src.model import CirrhosisPredictionPipeline  # noqa: E402
from src.predict import get_model_pipeline  # noqa: E402


@pytest.fixture(scope="session")
def model_pipeline() -> CirrhosisPredictionPipeline:
    """Session-scoped loaded model pipeline fixture."""
    return get_model_pipeline(settings.MODEL_PATH)


@pytest.fixture(scope="session")
def client() -> Generator[TestClient, None, None]:
    """TestClient fixture with app lifespan initialized."""
    app = create_app()
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def synthetic_healthy_patient() -> Dict[str, Any]:
    """Safe, synthetic clinical input simulating a low-risk profile."""
    return {
        "age": 35,
        "total_bilirubin": 0.8,
        "albumin": 4.1,
        "alt": 22.0,
        "ast": 24.0,
        "alp": 115.0,
        "total_proteins": 7.4,
    }


@pytest.fixture
def synthetic_high_risk_patient() -> Dict[str, Any]:
    """Safe, synthetic clinical input simulating an elevated cirrhosis risk profile."""
    return {
        "age": 63,
        "total_bilirubin": 8.5,
        "albumin": 2.2,
        "alt": 120.0,
        "ast": 160.0,
        "alp": 410.0,
        "total_proteins": 5.9,
    }


@pytest.fixture
def synthetic_patient_with_missing_values() -> Dict[str, Any]:
    """Safe, synthetic input with missing lab parameters to verify median imputation."""
    return {
        "age": 50,
        "total_bilirubin": 1.5,
        "albumin": 3.2,
        "alt": None,
        "ast": None,
        "alp": 210.0,
        "total_proteins": None,
    }


@pytest.fixture
def synthetic_invalid_patients() -> list:
    """Collection of invalid payloads for edge-case and boundary verification."""
    return [
        # Out-of-bounds negative age
        {
            "payload": {
                "age": -10,
                "total_bilirubin": 1.2,
                "albumin": 3.5,
                "alt": 30.0,
                "ast": 35.0,
                "alp": 150.0,
                "total_proteins": 7.0,
            },
            "desc": "Negative age",
        },
        # Impossibly high age
        {
            "payload": {
                "age": 150,
                "total_bilirubin": 1.2,
                "albumin": 3.5,
                "alt": 30.0,
                "ast": 35.0,
                "alp": 150.0,
                "total_proteins": 7.0,
            },
            "desc": "Age over 120",
        },
        # Negative bilirubin
        {
            "payload": {
                "age": 45,
                "total_bilirubin": -2.5,
                "albumin": 3.5,
                "alt": 30.0,
                "ast": 35.0,
                "alp": 150.0,
                "total_proteins": 7.0,
            },
            "desc": "Negative total bilirubin",
        },
        # Invalid data type (string instead of numeric)
        {
            "payload": {
                "age": "forty-five",
                "total_bilirubin": 1.2,
                "albumin": 3.5,
                "alt": 30.0,
                "ast": 35.0,
                "alp": 150.0,
                "total_proteins": 7.0,
            },
            "desc": "String value for age",
        },
        # Extra unknown forbidden field
        {
            "payload": {
                "age": 45,
                "total_bilirubin": 1.2,
                "albumin": 3.5,
                "alt": 30.0,
                "ast": 35.0,
                "alp": 150.0,
                "total_proteins": 7.0,
                "unrecognized_field": 999,
            },
            "desc": "Extra forbidden field",
        },
    ]
