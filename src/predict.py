"""
predict.py
----------
Loads the final saved model (and scaler) to score new, unseen
transactions — either a single transaction (as a dict) or a batch
(as a DataFrame/CSV). Used by both the Streamlit app and the
live transaction simulator.

Usage (CLI):
    python -m src.predict --csv data/new_transactions.csv
"""

import argparse
import logging
import joblib
import pandas as pd

from src import config

logging.basicConfig(level=config.LOG_LEVEL, format=config.LOG_FORMAT)
logger = logging.getLogger(__name__)

_model = None
_scaler = None


def _get_model():
    global _model
    if _model is None:
        logger.info(f"Loading model from {config.FINAL_MODEL_PATH}")
        _model = joblib.load(config.FINAL_MODEL_PATH)
    return _model


def _get_scaler():
    global _scaler
    if _scaler is None:
        logger.info(f"Loading scaler from {config.SCALER_PATH}")
        _scaler = joblib.load(config.SCALER_PATH)
    return _scaler


def preprocess_for_inference(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the same Amount/Time scaling used at training time."""
    df = df.copy()
    scaler = _get_scaler()
    cols_to_scale = [config.AMOUNT_COLUMN, config.TIME_COLUMN]
    df[cols_to_scale] = scaler.transform(df[cols_to_scale])
    # Ensure column order matches training feature order
    return df[config.ALL_FEATURE_COLUMNS]


def predict_transaction(transaction: dict) -> dict:
    """
    Predict fraud probability for a single transaction dict, e.g.:
        {"Time": 4062.0, "V1": -1.35, ..., "V28": 0.02, "Amount": 149.62}
    Returns: {"fraud_probability": float, "is_fraud": bool}
    """
    df = pd.DataFrame([transaction])
    df = preprocess_for_inference(df)

    model = _get_model()
    prob = float(model.predict_proba(df)[:, 1][0])
    is_fraud = prob >= config.DECISION_THRESHOLD

    return {"fraud_probability": prob, "is_fraud": bool(is_fraud)}


def predict_batch(df: pd.DataFrame) -> pd.DataFrame:
    """
    Predict fraud probability for a batch of transactions.
    Returns the input dataframe with two extra columns:
    'fraud_probability' and 'is_fraud'.
    """
    features = df[config.ALL_FEATURE_COLUMNS] if set(config.ALL_FEATURE_COLUMNS).issubset(df.columns) else df
    features = preprocess_for_inference(features)

    model = _get_model()
    probs = model.predict_proba(features)[:, 1]

    result = df.copy()
    result["fraud_probability"] = probs
    result["is_fraud"] = (probs >= config.DECISION_THRESHOLD)
    return result


def main():
    parser = argparse.ArgumentParser(description="Predict fraud on new transactions")
    parser.add_argument("--csv", type=str, required=True, help="Path to a CSV of new transactions")
    parser.add_argument("--out", type=str, default=None, help="Optional path to save predictions CSV")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    result = predict_batch(df)

    n_fraud = int(result["is_fraud"].sum())
    logger.info(f"Scored {len(result)} transactions -> {n_fraud} flagged as fraud")

    if args.out:
        result.to_csv(args.out, index=False)
        logger.info(f"Saved predictions to {args.out}")
    else:
        print(result.head(20))


if __name__ == "__main__":
    main()
