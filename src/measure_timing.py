"""
measure_timing.py
------------------
Measures and records the training/inference timings the supervisor
documentation explicitly asked for and flagged as "Not recorded":
  - Baseline Logistic Regression training time
  - Baseline Random Forest training time
  - GridSearchCV tuning time (matches the notebook 6 search space:
    n_estimators [100, 200], max_depth [10, 20], min_samples_split [2, 5],
    cv=3, scoring='f1')
  - Final (tuned) model inference time — total and per-transaction average

Uses the SAME preprocessing as the documented notebook pipeline
(no SMOTE, 80/20 split, random_state=42) so these numbers line up
with the metrics already reported in the supervisor documentation.
If you instead adopt the src/ pipeline (SMOTE + XGBoost) as the real
one, rerun this script after that decision — the timings below are
only valid for whichever pipeline actually produced final_model.pkl.

Usage (CLI):
    python -m src.measure_timing
    python -m src.measure_timing --out reports/timing_report.json
"""

import argparse
import json
import logging
import time

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

from src import config
from src.preprocess import load_raw_data, clean_data, scale_features, split_data

logging.basicConfig(level=config.LOG_LEVEL, format=config.LOG_FORMAT)
logger = logging.getLogger(__name__)

# Matches notebook 6's GridSearchCV search space exactly.
NOTEBOOK_RF_PARAM_GRID = {
    "n_estimators": [100, 200],
    "max_depth": [10, 20],
    "min_samples_split": [2, 5],
}


def _timed(fn, *args, **kwargs):
    """Run fn, return (result, elapsed_seconds)."""
    start = time.perf_counter()
    result = fn(*args, **kwargs)
    elapsed = time.perf_counter() - start
    return result, elapsed


def prepare_data():
    """Load + clean + scale + split, matching the notebook pipeline (no SMOTE)."""
    df = load_raw_data()
    df = clean_data(df)
    df, _scaler = scale_features(df, fit=True)
    X_train, X_test, y_train, y_test = split_data(df)
    return X_train, X_test, y_train, y_test


def measure_baseline_lr_training(X_train, y_train):
    model = LogisticRegression(max_iter=1000, random_state=config.RANDOM_STATE)
    _, elapsed = _timed(model.fit, X_train, y_train)
    logger.info(f"Baseline Logistic Regression training time: {elapsed:.4f}s")
    return elapsed


def measure_baseline_rf_training(X_train, y_train):
    model = RandomForestClassifier(class_weight="balanced", random_state=config.RANDOM_STATE)
    _, elapsed = _timed(model.fit, X_train, y_train)
    logger.info(f"Baseline Random Forest training time: {elapsed:.4f}s")
    return elapsed


def measure_gridsearch_tuning(X_train, y_train):
    base_model = RandomForestClassifier(class_weight="balanced", random_state=config.RANDOM_STATE)
    search = GridSearchCV(
        base_model,
        param_grid=NOTEBOOK_RF_PARAM_GRID,
        cv=3,
        scoring="f1",
        n_jobs=-1,
    )
    _, elapsed = _timed(search.fit, X_train, y_train)
    logger.info(
        f"GridSearchCV tuning time: {elapsed:.4f}s "
        f"(best params: {search.best_params_}, best CV F1: {search.best_score_:.4f})"
    )
    return elapsed, search.best_estimator_, search.best_params_, search.best_score_


def measure_inference_time(model, X_test):
    """Total inference time over the full test set, plus per-transaction average (ms)."""
    _, elapsed_total = _timed(model.predict, X_test)
    per_transaction_ms = (elapsed_total / len(X_test)) * 1000
    logger.info(
        f"Inference time: {elapsed_total:.4f}s total over {len(X_test)} transactions "
        f"({per_transaction_ms:.5f} ms/transaction)"
    )
    return elapsed_total, per_transaction_ms


def run_all(out_path=None):
    X_train, X_test, y_train, y_test = prepare_data()

    lr_time = measure_baseline_lr_training(X_train, y_train)
    rf_time = measure_baseline_rf_training(X_train, y_train)
    tuning_time, tuned_model, best_params, best_cv_f1 = measure_gridsearch_tuning(X_train, y_train)
    inference_total, inference_per_txn_ms = measure_inference_time(tuned_model, X_test)

    report = {
        "baseline_lr_training_seconds": round(lr_time, 4),
        "baseline_rf_training_seconds": round(rf_time, 4),
        "gridsearchcv_tuning_seconds": round(tuning_time, 4),
        "gridsearchcv_best_params": best_params,
        "gridsearchcv_best_cv_f1": round(best_cv_f1, 6),
        "final_model_inference_seconds_total": round(inference_total, 4),
        "final_model_inference_ms_per_transaction": round(inference_per_txn_ms, 5),
        "test_set_size": len(X_test),
        "note": (
            "SHAP explanation timing is not included here — measure it "
            "separately once explain.py is verified to run, e.g. by timing "
            "src.explain.explain_single_prediction over a sample of rows."
        ),
    }

    print(json.dumps(report, indent=2))

    if out_path:
        with open(out_path, "w") as f:
            json.dump(report, f, indent=2)
        logger.info(f"Saved timing report to {out_path}")

    return report


def main():
    parser = argparse.ArgumentParser(description="Measure FraudShield AI training/inference timing")
    parser.add_argument(
        "--out",
        type=str,
        default=str(config.REPORTS_DIR / "timing_report.json"),
        help="Path to save the JSON timing report",
    )
    args = parser.parse_args()
    run_all(out_path=args.out)


if __name__ == "__main__":
    main()
