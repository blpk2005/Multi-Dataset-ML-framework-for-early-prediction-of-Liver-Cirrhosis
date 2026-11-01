"""Application configuration module."""

from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


BASE_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """Application settings with environment variable support."""

    # Project metadata
    PROJECT_NAME: str = "Liver Cirrhosis Prediction API"
    VERSION: str = "1.0.0"
    API_V1_STR: str = "/api/v1"
    ENVIRONMENT: str = "development"
    LOG_LEVEL: str = "INFO"

    # Server settings
    HOST: str = "0.0.0.0"
    PORT: int = 8000

    # Model and artifacts
    ARTIFACTS_DIR: Path = BASE_DIR / "artifacts"
    MODEL_FILE_NAME: str = "model_pipeline.joblib"
    MODEL_PATH: Path = BASE_DIR / "artifacts" / "model_pipeline.joblib"

    # Inference settings
    DEFAULT_DECISION_THRESHOLD: float = 0.50
    RESEARCH_DISCLAIMER: str = (
        "RESEARCH NOTICE: Model outputs are statistical predictions generated "
        "for investigative and educational purposes only. They do not constitute "
        "medical advice, formal diagnosis, or treatment recommendations."
    )

    # Data paths for training/validation
    DATA_DIR: Path = BASE_DIR
    HEPATITIS_CSV: str = "HepatitisCdata.csv"
    LPD_TRAIN_CSV: str = "Liver Patient Dataset (LPD)_train (1) (1) (1).csv"
    INDIAN_TEST_CSV: str = "indian_liver_patient.csv"

    model_config = SettingsConfigDict(
        env_file=str(BASE_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()
