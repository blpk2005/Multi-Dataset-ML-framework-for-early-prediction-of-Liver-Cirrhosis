"""Unit tests for feature preprocessing, missing value imputation, and ordering."""

import pytest
import numpy as np
import pandas as pd

from src.preprocessing import (
    ClinicalFeaturePreprocessor,
    DEFAULT_FEATURE_NAMES,
)


def test_preprocessor_fitting_and_output_shape():
    """Verify preprocessor fits properly and generates expected (N, D) output."""
    raw_data = pd.DataFrame(
        {
            "Age": [25, 45, 65],
            "Total_Bilirubin": [0.8, 2.5, 5.0],
            "Albumin": [4.0, 3.2, 2.5],
            "ALT": [20.0, 45.0, 80.0],
            "AST": [22.0, 50.0, 95.0],
            "ALP": [100.0, 220.0, 350.0],
            "Total_Protiens": [7.5, 6.8, 5.9],
        }
    )

    preprocessor = ClinicalFeaturePreprocessor(feature_names=DEFAULT_FEATURE_NAMES)
    transformed = preprocessor.fit_transform(raw_data)

    assert isinstance(transformed, np.ndarray)
    assert transformed.shape == (3, len(DEFAULT_FEATURE_NAMES))
    assert not np.isnan(transformed).any()
    assert preprocessor.is_fitted


def test_feature_ordering_consistency():
    """Verify that feature ordering is strictly preserved regardless of input column order."""
    # Data with intentional inverted column order
    reversed_data = pd.DataFrame(
        {
            "Total_Protiens": [7.0],
            "ALP": [150.0],
            "AST": [30.0],
            "ALT": [35.0],
            "Albumin": [3.8],
            "Total_Bilirubin": [1.2],
            "Age": [50.0],
        }
    )

    standard_data = reversed_data[DEFAULT_FEATURE_NAMES]

    preprocessor = ClinicalFeaturePreprocessor()
    preprocessor.fit(standard_data)

    out_reversed = preprocessor.transform(reversed_data)
    out_standard = preprocessor.transform(standard_data)

    # Both must produce identical transformed vectors
    np.testing.assert_allclose(out_reversed, out_standard, rtol=1e-5)


def test_missing_value_imputation():
    """Verify that missing values (NaN) are replaced by fitted medians and scaled."""
    train_data = pd.DataFrame(
        {
            "Age": [30, 40, 50],
            "Total_Bilirubin": [1.0, 2.0, 3.0],  # Median is 2.0
            "Albumin": [3.0, 3.5, 4.0],
            "ALT": [20.0, 30.0, 40.0],
            "AST": [25.0, 35.0, 45.0],
            "ALP": [100.0, 150.0, 200.0],
            "Total_Protiens": [6.5, 7.0, 7.5],
        }
    )

    preprocessor = ClinicalFeaturePreprocessor()
    preprocessor.fit(train_data)

    # Incomplete test row with missing Total_Bilirubin
    incomplete_input = {
        "Age": 40,
        "Total_Bilirubin": None,
        "Albumin": 3.5,
        "ALT": 30.0,
        "AST": 35.0,
        "ALP": 150.0,
        "Total_Protiens": 7.0,
    }

    out = preprocessor.transform(incomplete_input)

    assert not np.isnan(out).any()
    # Feature at index 1 is Total_Bilirubin, median should have been 2.0, scaled to 0.0
    bilirubin_idx = DEFAULT_FEATURE_NAMES.index("Total_Bilirubin")
    assert abs(out[0, bilirubin_idx] - 0.0) < 1e-4


def test_unfitted_transform_raises_error():
    """Calling transform before fit must raise a descriptive RuntimeError."""
    preprocessor = ClinicalFeaturePreprocessor()
    with pytest.raises(RuntimeError, match="must be fitted"):
        preprocessor.transform({"Age": 45})


def test_alias_resolution():
    """Verify snake_case and clinical synonyms are mapped correctly."""
    preprocessor = ClinicalFeaturePreprocessor()
    train_df = pd.DataFrame({col: [10.0, 20.0] for col in DEFAULT_FEATURE_NAMES})
    preprocessor.fit(train_df)

    aliased_input = {
        "age": 45,
        "total_bilirubin": 1.2,
        "total_proteins": 6.8,
        "albumin": 3.5,
        "alt": 25.0,
        "ast": 30.0,
        "alp": 120.0,
    }
    result = preprocessor.transform(aliased_input)
    assert result.shape == (1, len(DEFAULT_FEATURE_NAMES))
    assert not np.isnan(result).any()


def test_preprocessor_summary():
    """Verify get_summary delivers diagnostic metadata."""
    preprocessor = ClinicalFeaturePreprocessor()
    train_df = pd.DataFrame({col: [1.0, 3.0, 5.0] for col in DEFAULT_FEATURE_NAMES})
    preprocessor.fit(train_df)

    summary = preprocessor.get_summary()
    assert summary["fitted"] is True
    assert len(summary["imputed_medians"]) == len(DEFAULT_FEATURE_NAMES)
    assert len(summary["scaler_means"]) == len(DEFAULT_FEATURE_NAMES)
