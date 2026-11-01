"""Inference service module for liver cirrhosis risk assessment."""

import logging
from typing import Dict, Any, List, Optional
from pathlib import Path

from src.config import settings
from src.model import CirrhosisPredictionPipeline

logger = logging.getLogger(__name__)

# Global cached model pipeline instance
_MODEL_PIPELINE: Optional[CirrhosisPredictionPipeline] = None


def get_model_pipeline(model_path: Optional[Path] = None) -> CirrhosisPredictionPipeline:
    """Retrieve or lazily initialize the singleton model pipeline."""
    global _MODEL_PIPELINE
    if _MODEL_PIPELINE is None:
        path = model_path or settings.MODEL_PATH
        logger.info("Initializing prediction pipeline from %s", path)
        _MODEL_PIPELINE = CirrhosisPredictionPipeline.load(path)
    return _MODEL_PIPELINE


def set_model_pipeline(pipeline: Optional[CirrhosisPredictionPipeline]) -> None:
    """Set or reset the in-memory pipeline (useful for testing or hot-reloading)."""
    global _MODEL_PIPELINE
    _MODEL_PIPELINE = pipeline


def is_model_loaded() -> bool:
    """Check if model pipeline is currently loaded in memory."""
    return _MODEL_PIPELINE is not None


def run_prediction(
    patient_data: Dict[str, Any],
    pipeline: Optional[CirrhosisPredictionPipeline] = None,
) -> Dict[str, Any]:
    """Execute prediction for a single patient record."""
    active_pipeline = pipeline or get_model_pipeline()
    detailed_results = active_pipeline.predict_detailed([patient_data])
    result = detailed_results[0]
    result["disclaimer"] = settings.RESEARCH_DISCLAIMER
    return result


def run_batch_prediction(
    patients_data: List[Dict[str, Any]],
    pipeline: Optional[CirrhosisPredictionPipeline] = None,
) -> Dict[str, Any]:
    """Execute predictions for a batch of clinical records."""
    active_pipeline = pipeline or get_model_pipeline()
    detailed_results = active_pipeline.predict_detailed(patients_data)
    for res in detailed_results:
        res["disclaimer"] = settings.RESEARCH_DISCLAIMER

    return {
        "count": len(detailed_results),
        "predictions": detailed_results,
        "disclaimer": settings.RESEARCH_DISCLAIMER,
    }
