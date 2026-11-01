"""Pydantic schemas for request validation and response serialization."""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, ConfigDict


class ClinicalInput(BaseModel):
    """Clinical lab parameters for liver disease / cirrhosis risk evaluation.

    Includes boundary validation based on physiological limits.
    All laboratory parameters accept missing values (None) which are
    handled through robust median imputation matching the training pipeline.
    """

    model_config = ConfigDict(
        populate_by_name=True,
        extra="forbid",
        json_schema_extra={
            "example": {
                "age": 55,
                "total_bilirubin": 1.8,
                "albumin": 3.4,
                "alt": 42.0,
                "ast": 39.0,
                "alp": 198.0,
                "total_proteins": 6.8,
            }
        },
    )

    age: Optional[float] = Field(
        default=None,
        ge=1,
        le=120,
        description="Patient age in years (1 - 120)",
        alias="Age",
    )
    total_bilirubin: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=100.0,
        description="Total Bilirubin in mg/dL (or normalized lab equivalent)",
        alias="Total_Bilirubin",
    )
    albumin: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=100.0,
        description="Serum Albumin in g/dL",
        alias="Albumin",
    )
    alt: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=3000.0,
        description="Alanine Aminotransferase (ALT) in U/L",
        alias="ALT",
    )
    ast: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=3000.0,
        description="Aspartate Aminotransferase (AST) in U/L",
        alias="AST",
    )
    alp: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=3000.0,
        description="Alkaline Phosphatase (ALP) in U/L",
        alias="ALP",
    )
    total_proteins: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=150.0,
        description="Total Proteins in g/dL",
        alias="Total_Protiens",
    )


class BatchPredictionRequest(BaseModel):
    """Batch prediction request payload."""

    patients: List[ClinicalInput] = Field(
        ...,
        min_length=1,
        max_length=500,
        description="List of clinical inputs for batch inference",
    )


class PredictionOutput(BaseModel):
    """Prediction outcome for a single clinical input."""

    prediction: int = Field(
        ...,
        description="Binary classification outcome: 1 indicates high risk, 0 indicates low risk",
    )
    prediction_label: str = Field(
        ...,
        description="Human-readable outcome ('High Risk' or 'Low Risk')",
    )
    cirrhosis_probability: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Estimated probability of liver disease / cirrhosis (0.0 to 1.0)",
    )
    decision_threshold: float = Field(
        ...,
        description="Calibrated decision threshold applied to probability",
    )
    risk_level: str = Field(
        ...,
        description="Risk category: 'Low', 'Moderate', 'High', or 'Critical'",
    )
    model_version: str = Field(
        ...,
        description="Version identifier of the deployed model pipeline",
    )
    disclaimer: str = Field(
        ...,
        description="Medical research disclaimer stating predictions are not clinical diagnoses",
    )


class BatchPredictionResponse(BaseModel):
    """Batch prediction response payload."""

    count: int = Field(..., description="Number of patients evaluated")
    predictions: List[PredictionOutput] = Field(
        ..., description="List of prediction outputs"
    )
    disclaimer: str = Field(..., description="Medical research disclaimer")


class HealthResponse(BaseModel):
    """Service health response."""

    status: str = Field("healthy", description="Current service health status")
    version: str = Field(..., description="API version")
    environment: str = Field(..., description="Deployment environment")


class ReadyResponse(BaseModel):
    """Service readiness response for Kubernetes / load balancers."""

    status: str = Field(..., description="'ready' or 'not_ready'")
    model_loaded: bool = Field(..., description="Whether the ML pipeline is loaded in memory")
    model_version: Optional[str] = Field(None, description="Active model version")
    feature_names: Optional[List[str]] = Field(None, description="Expected feature list")
    decision_threshold: Optional[float] = Field(
        None, description="Active decision threshold"
    )


class ModelInfoResponse(BaseModel):
    """Model metadata and lineage information."""

    model_name: str = Field(..., description="Model identifier")
    model_type: str = Field(..., description="Model architecture type (e.g. CatBoostClassifier)")
    version: str = Field(..., description="Model artifact version")
    feature_names: List[str] = Field(..., description="Ordered list of feature names")
    decision_threshold: float = Field(..., description="Optimized decision threshold")
    training_metrics: Dict[str, Any] = Field(
        default_factory=dict, description="Validation/test metrics recorded during training"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional training metadata, datasets used, and parameters"
    )
    disclaimer: str = Field(..., description="Medical research disclaimer")
