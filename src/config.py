"""
config.py
----------
Central configuration for the FraudShield AI project.
All paths, constants, and shared settings are defined here so that
every other module (preprocess, train_model, evaluate, predict,
explain, simulator) stays consistent and avoids hard-coded values.

NOTE: You mentioned config.py is already completed in your project.
This version is provided as a drop-in reference — replace values
below (paths, column names, thresholds) with your actual settings
if they differ.
"""

import os
from pathlib import Path

# ---------------------------------------------------------------------
# BASE PATHS
# ---------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent  # project root

DATA_DIR = BASE_DIR / "data"
RAW_DATA_PATH = DATA_DIR / "creditcard.csv"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
TRAIN_DATA_PATH = PROCESSED_DATA_DIR / "train.csv"
TEST_DATA_PATH = PROCESSED_DATA_DIR / "test.csv"

MODELS_DIR = BASE_DIR / "models"
FINAL_MODEL_PATH = MODELS_DIR / "final_model.pkl"
SCALER_PATH = MODELS_DIR / "scaler.pkl"

REPORTS_DIR = BASE_DIR / "reports"
METRICS_REPORT_PATH = REPORTS_DIR / "evaluation_metrics.json"
CONFUSION_MATRIX_PATH = REPORTS_DIR / "confusion_matrix.png"
ROC_CURVE_PATH = REPORTS_DIR / "roc_curve.png"
SHAP_SUMMARY_PATH = REPORTS_DIR / "shap_summary.png"

LOGS_DIR = BASE_DIR / "logs"

# Ensure key directories exist when config is imported
for _dir in [DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, REPORTS_DIR, LOGS_DIR]:
    os.makedirs(_dir, exist_ok=True)

# ---------------------------------------------------------------------
# DATASET SETTINGS
# ---------------------------------------------------------------------
# Standard Kaggle "Credit Card Fraud Detection" dataset schema:
# Time, V1-V28 (PCA components), Amount, Class (0 = legit, 1 = fraud)
TARGET_COLUMN = "Class"
AMOUNT_COLUMN = "Amount"
TIME_COLUMN = "Time"
FEATURE_COLUMNS_V = [f"V{i}" for i in range(1, 29)]
ALL_FEATURE_COLUMNS = [TIME_COLUMN] + FEATURE_COLUMNS_V + [AMOUNT_COLUMN]

# ---------------------------------------------------------------------
# TRAIN / TEST SPLIT
# ---------------------------------------------------------------------
TEST_SIZE = 0.2
RANDOM_STATE = 42

# ---------------------------------------------------------------------
# CLASS IMBALANCE HANDLING
# ---------------------------------------------------------------------
USE_SMOTE = True
SMOTE_SAMPLING_STRATEGY = 0.3   # minority : majority ratio after resampling

# ---------------------------------------------------------------------
# MODEL SETTINGS
# ---------------------------------------------------------------------
MODEL_CANDIDATES = ["logistic_regression", "random_forest", "xgboost"]
PRIMARY_METRIC = "f1"          # metric used to pick the best/final model
DECISION_THRESHOLD = 0.5       # probability cutoff for classifying fraud

RF_PARAMS = {
    "n_estimators": 300,
    "max_depth": 12,
    "min_samples_leaf": 2,
    "class_weight": "balanced",
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
}

XGB_PARAMS = {
    "n_estimators": 400,
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "eval_metric": "aucpr",
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
}

LOGREG_PARAMS = {
    "max_iter": 1000,
    "class_weight": "balanced",
    "random_state": RANDOM_STATE,
}

# Hyperparameter search grid (used by GridSearchCV / RandomizedSearchCV
# in train_model.py during tuning)
RF_PARAM_GRID = {
    "n_estimators": [200, 300, 500],
    "max_depth": [8, 12, 16, None],
    "min_samples_leaf": [1, 2, 4],
}

XGB_PARAM_GRID = {
    "n_estimators": [200, 400, 600],
    "max_depth": [4, 6, 8],
    "learning_rate": [0.01, 0.05, 0.1],
}

# ---------------------------------------------------------------------
# STREAMLIT / SIMULATOR SETTINGS
# ---------------------------------------------------------------------
SIMULATOR_STREAM_DELAY_SEC = 1.0     # delay between simulated transactions
SIMULATOR_BATCH_SIZE = 1
ALERT_PROBABILITY_THRESHOLD = 0.7    # probability above which a UI alert fires

# ---------------------------------------------------------------------
# LOGGING
# ---------------------------------------------------------------------
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
