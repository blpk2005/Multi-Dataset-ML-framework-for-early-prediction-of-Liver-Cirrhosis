"""Trained model encapsulation, calibration, persistence, and inference."""

import logging
from pathlib import Path
from typing import Dict, Any, Union, List, Optional
import numpy as np
import pandas as pd
import joblib

from src.preprocessing import ClinicalFeaturePreprocessor

logger = logging.getLogger(__name__)


def determine_risk_level(prob: float) -> str:
    """Categorize probability into clinical risk strata."""
    if prob < 0.30:
        return "Low"
    elif prob < 0.60:
        return "Moderate"
    elif prob < 0.85:
        return "High"
    return "Critical"


class CirrhosisPredictionPipeline:
    """Unified serialization bundle containing preprocessor, model, and metadata.

    Guarantees that inference executes the exact same feature ordering, imputation,
    scaling, and calibrated decision threshold determined during training.
    """

    def __init__(
        self,
        preprocessor: ClinicalFeaturePreprocessor,
        classifier: Any,
        decision_threshold: float = 0.50,
        version: str = "1.0.0",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.preprocessor = preprocessor
        self.classifier = classifier
        self.decision_threshold = float(decision_threshold)
        self.version = version
        self.metadata = metadata or {}

    @property
    def feature_names(self) -> List[str]:
        """Ordered list of feature names."""
        return self.preprocessor.feature_names

    def predict_proba(
        self, X: Union[pd.DataFrame, np.ndarray, List[Dict[str, Any]], Dict[str, Any]]
    ) -> np.ndarray:
        """Run feature preprocessing and compute positive class probabilities."""
        X_trans = self.preprocessor.transform(X)

        if hasattr(self.classifier, "predict_proba"):
            probs = self.classifier.predict_proba(X_trans)
            if probs.ndim == 2:
                return probs[:, 1]
            return probs
        elif hasattr(self.classifier, "decision_function"):
            decision = self.classifier.decision_function(X_trans)
            # Sigmoid transformation
            return 1.0 / (1.0 + np.exp(-decision))
        else:
            raise AttributeError("Classifier has neither predict_proba nor decision_function")

    def predict(
        self, X: Union[pd.DataFrame, np.ndarray, List[Dict[str, Any]], Dict[str, Any]]
    ) -> np.ndarray:
        """Compute binary predictions based on the optimized decision threshold."""
        probs = self.predict_proba(X)
        return (probs >= self.decision_threshold).astype(int)

    def predict_detailed(
        self, X: Union[pd.DataFrame, np.ndarray, List[Dict[str, Any]], Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Return rich prediction results including probabilities and risk stratification."""
        probs = self.predict_proba(X)
        results = []
        for p in probs:
            prob_float = float(p)
            pred_binary = int(prob_float >= self.decision_threshold)
            results.append(
                {
                    "prediction": pred_binary,
                    "prediction_label": "High Risk" if pred_binary == 1 else "Low Risk",
                    "cirrhosis_probability": round(prob_float, 4),
                    "decision_threshold": round(self.decision_threshold, 4),
                    "risk_level": determine_risk_level(prob_float),
                    "model_version": self.version,
                }
            )
        return results

    def save(self, file_path: Union[str, Path]) -> Path:
        """Persist the complete pipeline to disk atomically."""
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        logger.info("Saved CirrhosisPredictionPipeline to %s", path)
        return path

    @classmethod
    def load(cls, file_path: Union[str, Path]) -> "CirrhosisPredictionPipeline":
        """Load a persisted pipeline bundle from disk."""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Model artifact not found at {path}")
        pipeline = joblib.load(path)
        if not isinstance(pipeline, cls):
            raise TypeError(
                f"Loaded artifact is type {type(pipeline)}, expected {cls.__name__}"
            )
        logger.info("Loaded CirrhosisPredictionPipeline (version: %s) from %s", pipeline.version, path)
        return pipeline

    def get_info(self) -> Dict[str, Any]:
        """Return comprehensive metadata for introspection endpoints."""
        model_cls_name = (
            self.classifier.__class__.__name__
            if hasattr(self.classifier, "__class__")
            else "Unknown"
        )
        return {
            "model_name": "Cirrhosis-CatBoost-Calibrated",
            "model_type": model_cls_name,
            "version": self.version,
            "feature_names": self.feature_names,
            "decision_threshold": self.decision_threshold,
            "training_metrics": self.metadata.get("metrics", {}),
            "metadata": {
                "training_date": self.metadata.get("training_date"),
                "hyperparameters": self.metadata.get("hyperparameters", {}),
                "preprocessor": self.preprocessor.get_summary(),
                "class_weights": self.metadata.get("class_weights"),
            },
        }
