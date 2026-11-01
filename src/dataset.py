"""Dataset loading, cleaning, and harmonization module."""

import logging
from pathlib import Path
from typing import Tuple, List, Optional
import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import settings

logger = logging.getLogger(__name__)

COMMON_FEATURES: List[str] = [
    "Age",
    "Total_Bilirubin",
    "Albumin",
    "ALT",
    "AST",
    "ALP",
    "Total_Protiens",
]


def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Strip whitespace and replace spaces with underscores in DataFrame columns."""
    df = df.copy()
    df.columns = df.columns.str.strip().str.replace(" ", "_")
    return df


def load_raw_dataset(file_path: Path) -> pd.DataFrame:
    """Load a CSV file with error handling."""
    if not file_path.exists():
        raise FileNotFoundError(f"Dataset file not found at: {file_path}")
    logger.info("Loading dataset from %s", file_path)
    return pd.read_csv(file_path)


def harmonize_hepatitis(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize Hepatitis C dataset columns and binary target."""
    df = clean_column_names(df)
    if "Total_Proteins" in df.columns:
        df = df.rename(columns={"Total_Proteins": "Total_Protiens"})
    if "Target" not in df.columns:
        raise ValueError("Missing 'Target' column in Hepatitis C dataset")

    # Positive class (1): Cirrhosis/Hepatitis/Fibrosis; Negative (0): Blood Donor
    df["target"] = (df["Target"] != "0=Blood Donor").astype(int)
    missing_cols = [c for c in COMMON_FEATURES if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Hepatitis dataset missing expected features: {missing_cols}")
    return df[COMMON_FEATURES + ["target"]]


def harmonize_lpd_train(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize Liver Patient Dataset (LPD) train columns and binary target."""
    df = clean_column_names(df)
    if "Total_Proteins" in df.columns:
        df = df.rename(columns={"Total_Proteins": "Total_Protiens"})
    if "Result" not in df.columns:
        raise ValueError("Missing 'Result' column in LPD dataset")

    df["target"] = df["Result"].apply(lambda x: 1 if x == 1 else 0)
    missing_cols = [c for c in COMMON_FEATURES if c not in df.columns]
    if missing_cols:
        raise ValueError(f"LPD dataset missing expected features: {missing_cols}")
    return df[COMMON_FEATURES + ["target"]]


def harmonize_indian_patient(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize Indian Liver Patient Dataset test columns and binary target."""
    df = clean_column_names(df)
    if "Total_Proteins" in df.columns:
        df = df.rename(columns={"Total_Proteins": "Total_Protiens"})
    if "Dataset" not in df.columns:
        raise ValueError("Missing 'Dataset' column in Indian Liver Patient dataset")

    df["target"] = df["Dataset"].apply(lambda x: 1 if x == 1 else 0)
    missing_cols = [c for c in COMMON_FEATURES if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Indian patient dataset missing expected features: {missing_cols}")
    return df[COMMON_FEATURES + ["target"]]


def prepare_multi_dataset(
    data_dir: Optional[Path] = None,
    test_size: float = 0.20,
    random_state: int = 42,
    include_augmented_lpd: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load, standardize, deduplicate, and split clinical records into train and test sets.

    Note on data leakage prevention:
    The Kaggle LPD dataset (`Liver Patient Dataset (LPD)_train`) was artificially generated
    by replicating the 583 rows of the UCI Indian Liver Patient Dataset (`indian_liver_patient.csv`)
    ~50 times. Evaluating on `indian_liver_patient.csv` while training on `LPD_train` causes
    severe train-test overlap and artificial 100% accuracy.
    This function unifies genuine multi-dataset patient cohorts (Hepatitis C + ILPD),
    eliminates duplicate records, and performs an honest stratified train/test split.

    Returns:
        (train_df, test_df): Mutually exclusive train and test sets with realistic clinical variance.
    """
    directory = data_dir or settings.DATA_DIR

    hep_raw = load_raw_dataset(directory / settings.HEPATITIS_CSV)
    indian_raw = load_raw_dataset(directory / settings.INDIAN_TEST_CSV)

    hep = harmonize_hepatitis(hep_raw)
    indian = harmonize_indian_patient(indian_raw)

    datasets = [hep, indian]

    if include_augmented_lpd:
        lpd_raw = load_raw_dataset(directory / settings.LPD_TRAIN_CSV)
        lpd = harmonize_lpd_train(lpd_raw)
        datasets.append(lpd)

    # Combine and strictly deduplicate to guarantee zero leakage between train and test
    all_patients = pd.concat(datasets, axis=0, ignore_index=True)
    all_patients = all_patients.drop_duplicates().reset_index(drop=True)

    # Honest, stratified train/test split
    train_df, test_df = train_test_split(
        all_patients,
        test_size=test_size,
        stratify=all_patients["target"],
        random_state=random_state,
    )

    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    logger.info(
        "Prepared multi-dataset (leakage-free): %d unique patients -> %d train, %d test (positive rate: %.2f%%)",
        len(all_patients),
        len(train_df),
        len(test_df),
        100.0 * all_patients["target"].mean(),
    )
    return train_df, test_df
