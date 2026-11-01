"""FastAPI API routes for model serving, health checks, and metadata."""

import logging
import time
from fastapi import APIRouter, HTTPException, status
from prometheus_client import Counter, Histogram, Gauge

from src.config import settings
from src.predict import (
    get_model_pipeline,
    is_model_loaded,
    run_prediction,
    run_batch_prediction,
)
from api.schemas import (
    ClinicalInput,
    BatchPredictionRequest,
    PredictionOutput,
    BatchPredictionResponse,
    HealthResponse,
    ReadyResponse,
    ModelInfoResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter()

# Prometheus custom metrics
PREDICTION_REQUESTS = Counter(
    "cirrhosis_predictions_total",
    "Total number of clinical predictions evaluated",
    ["status", "risk_level"],
)
PREDICTION_LATENCY = Histogram(
    "cirrhosis_prediction_duration_seconds",
    "Time spent running prediction pipeline in seconds",
    buckets=[0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0],
)
MODEL_STATUS_GAUGE = Gauge(
    "cirrhosis_model_loaded",
    "1 if the machine learning model pipeline is loaded and ready, 0 otherwise",
)


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Service Health Check",
    tags=["Health & Status"],
)
async def health_check() -> HealthResponse:
    """Liveness probe: verifies that the HTTP service is running."""
    return HealthResponse(
        status="healthy",
        version=settings.VERSION,
        environment=settings.ENVIRONMENT,
    )


@router.get(
    "/ready",
    response_model=ReadyResponse,
    summary="Readiness Probe",
    tags=["Health & Status"],
)
async def readiness_check() -> ReadyResponse:
    """Readiness probe: verifies that the ML model bundle is loaded and can serve requests."""
    loaded = is_model_loaded()
    if not loaded:
        try:
            get_model_pipeline()
            loaded = True
        except Exception as exc:
            logger.warning("Readiness probe failed: model not ready (%s)", exc)
            MODEL_STATUS_GAUGE.set(0)
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"Model pipeline not loaded or unavailable: {exc}",
            )

    MODEL_STATUS_GAUGE.set(1)
    pipeline = get_model_pipeline()
    return ReadyResponse(
        status="ready",
        model_loaded=True,
        model_version=pipeline.version,
        feature_names=pipeline.feature_names,
        decision_threshold=pipeline.decision_threshold,
    )


@router.post(
    "/predict",
    response_model=PredictionOutput,
    summary="Predict Cirrhosis Risk",
    tags=["Inference"],
)
async def predict_cirrhosis(payload: ClinicalInput) -> PredictionOutput:
    """Validate clinical lab inputs and generate cirrhosis / liver disease risk prediction.

    Missing clinical fields are imputed using median values established during training.
    """
    start_time = time.perf_counter()
    try:
        pipeline = get_model_pipeline()
    except Exception as exc:
        PREDICTION_REQUESTS.labels(status="error", risk_level="none").inc()
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Inference model is currently unavailable: {exc}",
        )

    # Convert Pydantic payload to dictionary (excluding None if desired, or passing through)
    input_dict = payload.model_dump(by_alias=True)

    try:
        result = run_prediction(input_dict, pipeline=pipeline)
        elapsed = time.perf_counter() - start_time
        PREDICTION_LATENCY.observe(elapsed)
        PREDICTION_REQUESTS.labels(
            status="success", risk_level=result["risk_level"]
        ).inc()
        return PredictionOutput(**result)
    except Exception as exc:
        PREDICTION_REQUESTS.labels(status="error", risk_level="none").inc()
        logger.exception("Prediction failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal inference failure: {str(exc)}",
        )


@router.post(
    "/predict/batch",
    response_model=BatchPredictionResponse,
    summary="Batch Predict Cirrhosis Risk",
    tags=["Inference"],
)
async def predict_cirrhosis_batch(
    payload: BatchPredictionRequest,
) -> BatchPredictionResponse:
    """Run batch inference for multiple patient records."""
    try:
        pipeline = get_model_pipeline()
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Inference model is currently unavailable: {exc}",
        )

    patient_records = [p.model_dump(by_alias=True) for p in payload.patients]
    try:
        batch_res = run_batch_prediction(patient_records, pipeline=pipeline)
        return BatchPredictionResponse(**batch_res)
    except Exception as exc:
        logger.exception("Batch prediction failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal inference failure: {str(exc)}",
        )


@router.get(
    "/model-info",
    response_model=ModelInfoResponse,
    summary="Model Metadata & Provenance",
    tags=["Metadata"],
)
async def model_info() -> ModelInfoResponse:
    """Retrieve detailed model lineage, version, hyperparameters, and training metrics."""
    try:
        pipeline = get_model_pipeline()
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Model artifact not accessible: {exc}",
        )

    info = pipeline.get_info()
    info["disclaimer"] = settings.RESEARCH_DISCLAIMER
    return ModelInfoResponse(**info)
