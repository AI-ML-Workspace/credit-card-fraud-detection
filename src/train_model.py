"""
train_model.py
---------------
Trains candidate models (Logistic Regression, Random Forest, XGBoost),
optionally tunes hyperparameters, compares them on validation metrics,
and saves the best-performing model as models/final_model.pkl.

Usage (CLI):
    python -m src.train_model                # train all candidates
    python -m src.train_model --tune          # also run hyperparameter search
    python -m src.train_model --model xgboost # train a single model only
"""

import argparse
import logging
import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import f1_score, roc_auc_score

from src import config
from src.preprocess import run_pipeline

logging.basicConfig(level=config.LOG_LEVEL, format=config.LOG_FORMAT)
logger = logging.getLogger(__name__)


def get_model_instance(name: str):
    """Return an untrained model instance for the given model name."""
    if name == "logistic_regression":
        return LogisticRegression(**config.LOGREG_PARAMS)
    elif name == "random_forest":
        return RandomForestClassifier(**config.RF_PARAMS)
    elif name == "xgboost":
        from xgboost import XGBClassifier
        return XGBClassifier(**config.XGB_PARAMS)
    else:
        raise ValueError(f"Unknown model name: {name}")


def get_param_grid(name: str):
    if name == "random_forest":
        return config.RF_PARAM_GRID
    elif name == "xgboost":
        return config.XGB_PARAM_GRID
    return None


def tune_model(name: str, X_train, y_train):
    """Run RandomizedSearchCV for models that have a defined param grid."""
    grid = get_param_grid(name)
    base_model = get_model_instance(name)

    if grid is None:
        logger.info(f"No tuning grid defined for {name}; using default params.")
        base_model.fit(X_train, y_train)
        return base_model

    logger.info(f"Running RandomizedSearchCV for {name}...")
    search = RandomizedSearchCV(
        base_model,
        param_distributions=grid,
        n_iter=8,
        scoring="f1",
        cv=3,
        random_state=config.RANDOM_STATE,
        n_jobs=-1,
        verbose=1,
    )
    search.fit(X_train, y_train)
    logger.info(f"Best params for {name}: {search.best_params_}")
    return search.best_estimator_


def train_candidates(X_train, y_train, X_val, y_val, tune: bool = False, only_model: str = None):
    """Train each candidate model and score it on the validation set."""
    candidates = [only_model] if only_model else config.MODEL_CANDIDATES
    results = {}

    for name in candidates:
        logger.info(f"--- Training {name} ---")
        if tune:
            model = tune_model(name, X_train, y_train)
        else:
            model = get_model_instance(name)
            model.fit(X_train, y_train)

        val_probs = model.predict_proba(X_val)[:, 1]
        val_preds = (val_probs >= config.DECISION_THRESHOLD).astype(int)

        f1 = f1_score(y_val, val_preds)
        auc = roc_auc_score(y_val, val_probs)
        logger.info(f"{name} -> F1: {f1:.4f} | ROC-AUC: {auc:.4f}")

        results[name] = {"model": model, "f1": f1, "roc_auc": auc}

    return results


def select_best_model(results: dict):
    """Pick the model with the highest score on config.PRIMARY_METRIC."""
    best_name = max(results, key=lambda k: results[k][config.PRIMARY_METRIC])
    best_model = results[best_name]["model"]
    logger.info(
        f"Selected best model: {best_name} "
        f"({config.PRIMARY_METRIC}={results[best_name][config.PRIMARY_METRIC]:.4f})"
    )
    return best_name, best_model


def save_model(model, path=config.FINAL_MODEL_PATH):
    joblib.dump(model, path)
    logger.info(f"Saved final model to {path}")


def main():
    parser = argparse.ArgumentParser(description="Train FraudShield AI models")
    parser.add_argument("--tune", action="store_true", help="Run hyperparameter tuning")
    parser.add_argument("--model", type=str, default=None,
                         help="Train only a single model (logistic_regression | random_forest | xgboost)")
    args = parser.parse_args()

    X_train, X_test, y_train, y_test, _ = run_pipeline(save=True)

    results = train_candidates(
        X_train, y_train, X_test, y_test,
        tune=args.tune, only_model=args.model,
    )

    if args.model:
        # If training a single model explicitly, save it directly.
        save_model(results[args.model]["model"])
    else:
        best_name, best_model = select_best_model(results)
        save_model(best_model)

    summary = pd.DataFrame({
        name: {"f1": r["f1"], "roc_auc": r["roc_auc"]} for name, r in results.items()
    }).T
    logger.info(f"\nModel comparison summary:\n{summary}")


if __name__ == "__main__":
    main()
