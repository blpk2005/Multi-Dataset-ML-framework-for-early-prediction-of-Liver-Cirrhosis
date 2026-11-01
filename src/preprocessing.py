"""Clinical feature preprocessing and transformation pipeline."""

import logging
from typing import List, Dict, Any, Union, Optional
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

DEFAULT_FEATURE_NAMES: List[str] = [
    "Age",
    "Total_Bilirubin",
    "Albumin",
    "ALT",
    "AST",
    "ALP",
    "Total_Protiens",
]

# Field mapping for clinical input variations
FEATURE_ALIASES: Dict[str, str] = {
    "age": "Age",
    "total_bilirubin": "Total_Bilirubin",
    "total_proteins": "Total_Protiens",
    "total_protiens": "Total_Protiens",
    "albumin": "Albumin",
    "alt": "ALT",
    "ast": "AST",
    "alp": "ALP",
}


class ClinicalFeaturePreprocessor(BaseEstimator, TransformerMixin):
    """Robust preprocessing pipeline for clinical liver disease tabular data.

    Maintains deterministic feature ordering, handles missing clinical data
    via median imputation, and applies standard feature scaling.
    """

    def __init__(
        self,
        feature_names: Optional[List[str]] = None,
        imputer_strategy: str = "median",
    ) -> None:
        self.feature_names: List[str] = feature_names or list(DEFAULT_FEATURE_NAMES)
        self.imputer_strategy: str = imputer_strategy
        self.imputer: SimpleImputer = SimpleImputer(strategy=self.imputer_strategy)
        self.scaler: StandardScaler = StandardScaler()
        self.is_fitted: bool = False

    def _standardize_dataframe(
        self, X: Union[pd.DataFrame, List[Dict[str, Any]], Dict[str, Any]]
    ) -> pd.DataFrame:
        """Convert varied inputs into a DataFrame adhering to strict feature columns."""
        if isinstance(X, dict):
            df = pd.DataFrame([X])
        elif isinstance(X, list):
            df = pd.DataFrame(X)
        elif isinstance(X, pd.DataFrame):
            df = X.copy()
        else:
            raise TypeError(f"Unsupported data type for preprocessing: {type(X)}")

        # Rename aliases to standard feature names
        rename_map = {}
        for col in df.columns:
            cleaned = str(col).strip().replace(" ", "_")
            lower = cleaned.lower()
            if cleaned in self.feature_names:
                rename_map[col] = cleaned
            elif lower in FEATURE_ALIASES:
                rename_map[col] = FEATURE_ALIASES[lower]
        df = df.rename(columns=rename_map)

        # Ensure all expected columns exist, fill absent ones with NaN for imputation
        for feature in self.feature_names:
            if feature not in df.columns:
                df[feature] = np.nan

        # Enforce exact column order
        df = df[self.feature_names]

        # Convert to float64
        for feature in self.feature_names:
            df[feature] = pd.to_numeric(df[feature], errors="coerce")

        return df

    def fit(
        self, X: Union[pd.DataFrame, np.ndarray], y: Optional[Any] = None
    ) -> "ClinicalFeaturePreprocessor":
        """Fit imputer and scaler on training data."""
        if isinstance(X, pd.DataFrame):
            df = self._standardize_dataframe(X)
            values = df.to_numpy(dtype=np.float64)
        elif isinstance(X, np.ndarray):
            if X.shape[1] != len(self.feature_names):
                raise ValueError(
                    f"Array has {X.shape[1]} features, expected {len(self.feature_names)}"
                )
            values = X.astype(np.float64)
        else:
            df = self._standardize_dataframe(X)
            values = df.to_numpy(dtype=np.float64)

        imputed = self.imputer.fit_transform(values)
        self.scaler.fit(imputed)
        self.is_fitted = True
        logger.info("Fitted ClinicalFeaturePreprocessor with %d features", len(self.feature_names))
        return self

    def transform(
        self, X: Union[pd.DataFrame, np.ndarray, List[Dict[str, Any]], Dict[str, Any]]
    ) -> np.ndarray:
        """Transform clinical inputs using fitted imputer and scaler."""
        if not self.is_fitted:
            raise RuntimeError("ClinicalFeaturePreprocessor must be fitted before transforming data.")

        if isinstance(X, np.ndarray):
            if X.shape[1] != len(self.feature_names):
                raise ValueError(
                    f"Array shape {X.shape} does not match expected {len(self.feature_names)} features"
                )
            values = X.astype(np.float64)
        else:
            df = self._standardize_dataframe(X)
            values = df.to_numpy(dtype=np.float64)

        imputed = self.imputer.transform(values)
        scaled = self.scaler.transform(imputed)
        return scaled

    def fit_transform(
        self, X: Union[pd.DataFrame, np.ndarray], y: Optional[Any] = None
    ) -> np.ndarray:
        """Fit and transform in a single call."""
        return self.fit(X, y).transform(X)

    def get_summary(self) -> Dict[str, Any]:
        """Return preprocessor statistics for metadata and diagnostics."""
        if not self.is_fitted:
            return {"fitted": False}

        return {
            "fitted": True,
            "feature_names": self.feature_names,
            "imputed_medians": {
                name: float(med)
                for name, med in zip(self.feature_names, self.imputer.statistics_)
            },
            "scaler_means": {
                name: float(m)
                for name, m in zip(self.feature_names, self.scaler.mean_)
            },
            "scaler_scales": {
                name: float(s)
                for name, s in zip(self.feature_names, self.scaler.scale_)
            },
        }
