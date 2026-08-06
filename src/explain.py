"""
explain.py
----------
Explainable AI (XAI) module using SHAP. Provides:
  - Global feature importance (summary plot) across the test set
  - Local, per-transaction explanations (which features pushed a
    single prediction toward "fraud" vs "legit")

Used by the Streamlit dashboard (Phase 4) to show *why* a
transaction was flagged, not just that it was flagged.

Usage (CLI):
    python -m src.explain --global-summary
    python -m src.explain --explain-index 5
"""

import argparse
import logging
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap

from src import config

logging.basicConfig(level=config.LOG_LEVEL, format=config.LOG_FORMAT)
logger = logging.getLogger(__name__)

_explainer = None


def load_model_and_data():
    model = joblib.load(config.FINAL_MODEL_PATH)
    test_df = pd.read_csv(config.TEST_DATA_PATH)
    X_test = test_df.drop(columns=[config.TARGET_COLUMN])
    return model, X_test


def get_explainer(model, background_data: pd.DataFrame):
    """
    Build (and cache) a SHAP explainer.
    TreeExplainer is used for tree-based models (RandomForest/XGBoost);
    falls back to a generic Explainer (KernelExplainer-based) for
    linear models like Logistic Regression.
    """
    global _explainer
    if _explainer is not None:
        return _explainer

    model_type = type(model).__name__.lower()
    if "forest" in model_type or "xgb" in model_type or "tree" in model_type:
        logger.info("Using SHAP TreeExplainer")
        _explainer = shap.TreeExplainer(model)
    else:
        logger.info("Using generic SHAP Explainer (model-agnostic)")
        sample_bg = background_data.sample(min(100, len(background_data)),
                                            random_state=config.RANDOM_STATE)
        _explainer = shap.Explainer(model.predict_proba, sample_bg)

    return _explainer


def compute_shap_values(model, X: pd.DataFrame):
    explainer = get_explainer(model, X)
    shap_values = explainer.shap_values(X) if hasattr(explainer, "shap_values") else explainer(X)
    return shap_values


def plot_global_summary(sample_size: int = 500, save_path=config.SHAP_SUMMARY_PATH):
    """Generate a SHAP summary (beeswarm) plot over a sample of the test set."""
    model, X_test = load_model_and_data()
    sample = X_test.sample(min(sample_size, len(X_test)), random_state=config.RANDOM_STATE)

    explainer = get_explainer(model, X_test)
    shap_values = explainer.shap_values(sample) if hasattr(explainer, "shap_values") else explainer(sample)

    # For binary classifiers, TreeExplainer may return a list [class0, class1]
    values_for_fraud_class = shap_values[1] if isinstance(shap_values, list) else shap_values

    plt.figure()
    shap.summary_plot(values_for_fraud_class, sample, show=False)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved SHAP global summary plot to {save_path}")
    return save_path


def explain_single_prediction(transaction_row: pd.DataFrame) -> pd.DataFrame:
    """
    Return the top contributing features (with SHAP values) for a
    single transaction. `transaction_row` must be a 1-row, already
    scaled/preprocessed DataFrame matching training feature order
    (see predict.preprocess_for_inference).
    """
    model, X_test = load_model_and_data()
    explainer = get_explainer(model, X_test)

    shap_values = explainer.shap_values(transaction_row) if hasattr(explainer, "shap_values") \
        else explainer(transaction_row)

    values = shap_values[1][0] if isinstance(shap_values, list) else np.array(shap_values)[0]

    contrib = pd.DataFrame({
        "feature": transaction_row.columns,
        "value": transaction_row.iloc[0].values,
        "shap_value": values,
    })
    contrib["abs_impact"] = contrib["shap_value"].abs()
    contrib = contrib.sort_values("abs_impact", ascending=False).drop(columns="abs_impact")
    return contrib.reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser(description="SHAP explainability for FraudShield AI")
    parser.add_argument("--global-summary", action="store_true", help="Generate global SHAP summary plot")
    parser.add_argument("--explain-index", type=int, default=None,
                         help="Row index in the test set to explain individually")
    args = parser.parse_args()

    if args.global_summary:
        plot_global_summary()

    if args.explain_index is not None:
        _, X_test = load_model_and_data()
        row = X_test.iloc[[args.explain_index]]
        contrib = explain_single_prediction(row)
        print(contrib.head(10))


if __name__ == "__main__":
    main()
