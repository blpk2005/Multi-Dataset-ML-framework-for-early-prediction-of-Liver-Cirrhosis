# Multi-Dataset-ML & DL -framework-for-early-prediction-of-Liver-Cirrhosis

A comprehensive machine learning framework for the early prediction of liver cirrhosis using multiple clinical datasets. The project integrates advanced preprocessing, hyperparameter optimization, probability calibration, explainable AI, and cross-dataset benchmarking to identify robust and clinically interpretable predictive models.

---

## Project Overview

Liver cirrhosis is a progressive and potentially life-threatening disease caused by chronic liver damage. Early detection is critical for improving treatment outcomes and reducing complications.

This project presents a unified machine learning framework that evaluates multiple datasets and state-of-the-art models to predict liver cirrhosis at an early stage. The framework emphasizes:

- High predictive performance
- Model calibration
- Clinical interpretability
- Cross-dataset generalization
- Explainable AI using SHAP

---

## Key Features

- Multi-dataset evaluation using public liver disease datasets
- Automated preprocessing pipeline
- Missing value imputation
- Feature scaling and encoding
- Class imbalance handling
- Hyperparameter optimization with Optuna
- Probability calibration using Isotonic Regression and Platt Scaling
- Threshold optimization based on F1 score
- Explainability using SHAP
- Comparative benchmarking of traditional ML and deep learning models

---

## Datasets Used

### Indian Liver Patient Dataset (ILPD)
- Binary classification: Liver Disease vs No Liver Disease

### Hepatitis C Dataset
- Multi-class classification for liver condition analysis

### Cirrhosis Prediction Dataset
- Clinical dataset for predicting cirrhosis severity and outcomes

The framework is designed to be easily extended to additional medical datasets.

---

## Models Implemented

### Machine Learning Models
- CatBoost

### Deep Learning Models
- TabNet

---

## Advanced Techniques

### Hyperparameter Optimization
Optuna is used to automatically search for optimal model configurations.

### Probalistic Calibration
The reliability of the predicited models. Brier score and calibration curves are used to show the reliability of our model 

### Threshold Optimization
Decision thresholds are tuned to maximize F1 score rather than relying solely on the default threshold of 0.5.

### Explainable AI
SHAP (SHapley Additive exPlanations) is used to:
- Rank feature importance
- Visualize feature effects
- Interpret individual predictions

---

## Evaluation Metrics

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC
- PR-AUC
- Brier Score
- Calibration Curves
- Confusion Matrix

---

## Project Architecture

```text
Raw Clinical Datasets
        ↓
Data Cleaning & Preprocessing
        ↓
Imputation / Encoding / Scaling
        ↓
Train-Test Split
        ↓
Class Imbalance Handling
        ↓
Hyperparameter Optimization (Optuna)
        ↓
Model Training
        ↓
Probability Calibration
        ↓
Threshold Optimization
        ↓
Model Evaluation
        ↓
SHAP Explainability
        ↓
Cross-Dataset Benchmarking
