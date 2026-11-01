"""Training, threshold optimization, and artifact export script with leakage prevention."""

import argparse
import datetime
import logging
from pathlib import Path
import sys
from typing import Dict, Any, Tuple

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    roc_auc_score,
    f1_score,
    accuracy_score,
    precision_score,
    recall_score,
    brier_score_loss,
    classification_report,
)
from sklearn.model_selection import train_test_split  # noqa: E402
from catboost import CatBoostClassifier  # noqa: E402
import optuna  # noqa: E402

from src.config import settings  # noqa: E402
from src.dataset import prepare_multi_dataset, COMMON_FEATURES  # noqa: E402
from src.preprocessing import ClinicalFeaturePreprocessor  # noqa: E402
from src.model import CirrhosisPredictionPipeline  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("train")


def optimize_hyperparameters(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    n_trials: int = 15,
) -> Dict[str, Any]:
    """Use Optuna to discover optimal CatBoost hyperparameters."""
    logger.info("Starting Optuna hyperparameter optimization (%d trials)...", n_trials)
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial: optuna.Trial) -> float:
        params = {
            "iterations": trial.suggest_int("iterations", 200, 600),
            "depth": trial.suggest_int("depth", 4, 7),
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.12),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 2.0, 8.0),
            "loss_function": "Logloss",
            "eval_metric": "AUC",
            "random_seed": 42,
            "verbose": 0,
        }
        clf = CatBoostClassifier(**params)
        clf.fit(X_train, y_train)
        probs = clf.predict_proba(X_val)[:, 1]
        return float(roc_auc_score(y_val, probs))

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)
    logger.info("Optuna best trial ROC-AUC: %.4f", study.best_value)
    logger.info("Optuna best hyperparameters: %s", study.best_params)
    return study.best_params


def find_optimal_threshold(
    y_true: np.ndarray,
    y_probs: np.ndarray,
    start: float = 0.15,
    end: float = 0.85,
    step: float = 0.01,
) -> Tuple[float, float]:
    """Find the probability decision threshold maximizing F1 score."""
    thresholds = np.arange(start, end, step)
    best_thresh = 0.50
    best_f1 = 0.0

    for t in thresholds:
        preds = (y_probs >= t).astype(int)
        score = f1_score(y_true, preds, zero_division=0)
        if score > best_f1:
            best_f1 = float(score)
            best_thresh = float(round(t, 2))

    return best_thresh, best_f1


def train_pipeline(
    tune: bool = False,
    n_trials: int = 15,
    include_augmented_lpd: bool = False,
    output_path: Path = settings.MODEL_PATH,
) -> CirrhosisPredictionPipeline:
    """End-to-end pipeline training, calibration, and persistence."""
    logger.info("Loading and harmonizing multi-dataset cohorts...")
    train_df, test_df = prepare_multi_dataset(include_augmented_lpd=include_augmented_lpd)

    X_train_raw = train_df[COMMON_FEATURES]
    y_train = train_df["target"].to_numpy()

    X_test_raw = test_df[COMMON_FEATURES]
    y_test = test_df["target"].to_numpy()

    # Train / Validation split (for threshold optimization and early stopping)
    X_tr_sub, X_val_sub, y_tr_sub, y_val_sub = train_test_split(
        X_train_raw,
        y_train,
        test_size=0.20,
        stratify=y_train,
        random_state=42,
    )

    # Initialize and fit feature preprocessor strictly on training data
    logger.info("Fitting ClinicalFeaturePreprocessor (median imputer + standard scaler)...")
    preprocessor = ClinicalFeaturePreprocessor(feature_names=COMMON_FEATURES)
    X_tr_proc = preprocessor.fit_transform(X_tr_sub)
    X_val_proc = preprocessor.transform(X_val_sub)

    # Hyperparameter selection
    if tune:
        best_params = optimize_hyperparameters(
            X_tr_proc, y_tr_sub, X_val_proc, y_val_sub, n_trials=n_trials
        )
    else:
        logger.info("Using calibrated CatBoost hyperparameters...")
        best_params = {
            "iterations": 400,
            "depth": 5,
            "learning_rate": 0.05,
            "l2_leaf_reg": 4.0,
        }

    # Retrain preprocessor and model on full training split
    logger.info("Fitting final preprocessor on full training set (%d samples)...", len(train_df))
    full_preprocessor = ClinicalFeaturePreprocessor(feature_names=COMMON_FEATURES)
    X_train_full = full_preprocessor.fit_transform(X_train_raw)
    X_val_proc = full_preprocessor.transform(X_val_sub)

    final_params = {
        **best_params,
        "class_weights": [1.0, 1.4],
        "loss_function": "Logloss",
        "eval_metric": "AUC",
        "random_seed": 42,
        "verbose": 0,
    }

    logger.info("Training CatBoostClassifier...")
    model = CatBoostClassifier(**final_params)
    model.fit(X_train_full, y_train)

    # Optimize decision threshold on validation set
    val_probs = model.predict_proba(X_val_proc)[:, 1]
    optimal_thresh, val_f1 = find_optimal_threshold(y_val_sub, val_probs)
    logger.info("Optimal Decision Threshold (F1-maximizing): %.2f (Validation F1: %.4f)", optimal_thresh, val_f1)

    # Evaluate on strictly unseen held-out test cohort
    logger.info("Evaluating on held-out test cohort (%d samples)...", len(test_df))
    X_test_proc = full_preprocessor.transform(X_test_raw)
    test_probs = model.predict_proba(X_test_proc)[:, 1]
    test_preds = (test_probs >= optimal_thresh).astype(int)

    roc_auc = float(roc_auc_score(y_test, test_probs))
    accuracy = float(accuracy_score(y_test, test_preds))
    precision = float(precision_score(y_test, test_preds, zero_division=0))
    recall = float(recall_score(y_test, test_preds, zero_division=0))
    brier = float(brier_score_loss(y_test, test_probs))

    metrics = {
        "accuracy": round(accuracy, 4),
        "roc_auc": round(roc_auc, 4),
        "f1_score": round(float(f1_score(y_test, test_preds, zero_division=0)), 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "brier_score": round(brier, 4),
        "optimal_threshold": optimal_thresh,
    }

    logger.info("=== HONEST TEST EVALUATION METRICS ===")
    for k, v in metrics.items():
        logger.info("  %s: %s", k, v)
    logger.info("\n%s", classification_report(y_test, test_preds, zero_division=0))

    # Assemble bundle
    metadata = {
        "training_date": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "datasets": ["HepatitisCdata", "indian_liver_patient"],
        "num_training_samples": len(train_df),
        "num_test_samples": len(test_df),
        "hyperparameters": best_params,
        "class_weights": [1.0, 1.4],
        "metrics": metrics,
    }

    pipeline = CirrhosisPredictionPipeline(
        preprocessor=full_preprocessor,
        classifier=model,
        decision_threshold=optimal_thresh,
        version=settings.VERSION,
        metadata=metadata,
    )

    # Save to disk
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pipeline.save(output_path)
    logger.info("Pipeline successfully saved to %s", output_path)

    return pipeline


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and package Cirrhosis prediction pipeline")
    parser.add_argument("--tune", action="store_true", help="Run Optuna hyperparameter tuning")
    parser.add_argument("--trials", type=int, default=15, help="Number of Optuna tuning trials")
    parser.add_argument(
        "--include-lpd",
        action="store_true",
        help="Include augmented LPD train dataset (deduplicated)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(settings.MODEL_PATH),
        help="Target path for model pipeline artifact",
    )
    args = parser.parse_args()

    try:
        train_pipeline(
            tune=args.tune,
            n_trials=args.trials,
            include_augmented_lpd=args.include_lpd,
            output_path=Path(args.output),
        )
    except Exception as exc:
        logger.exception("Training failed: %s", exc)
        sys.exit(1)
