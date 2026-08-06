"""
simulator.py
------------
Live Transaction Simulator (Phase 5). Streams transactions one at a
time (from the held-out test set, so ground-truth labels exist for
demo/validation purposes), scores each with the final model, and
yields a result dict suitable for a live-updating Streamlit table
or dashboard feed.

Can also be run standalone from the CLI to print a live feed to the
console, useful for testing before wiring it into the Streamlit app.

Usage (CLI):
    python -m src.simulator --n 50 --delay 0.5
"""

import argparse
import logging
import time
from datetime import datetime

import pandas as pd

from src import config
from src.predict import predict_batch

logging.basicConfig(level=config.LOG_LEVEL, format=config.LOG_FORMAT)
logger = logging.getLogger(__name__)


def load_stream_source(shuffle: bool = True) -> pd.DataFrame:
    """
    Load the pool of transactions to stream from. Uses the test set
    (held out from training) so labels are available for comparison,
    but predictions are made blind to the label.
    """
    df = pd.read_csv(config.TEST_DATA_PATH)
    if shuffle:
        df = df.sample(frac=1, random_state=None).reset_index(drop=True)
    return df


def stream_transactions(n: int = None, delay: float = config.SIMULATOR_STREAM_DELAY_SEC,
                         shuffle: bool = True):
    """
    Generator that yields one scored transaction at a time.

    Each yielded item is a dict:
        {
            "timestamp": str,
            "transaction": dict (original feature values),
            "true_label": int (0/1, only present because we're using
                                labeled test data for the demo),
            "fraud_probability": float,
            "is_fraud_predicted": bool,
            "alert": bool  # True if probability crosses the alert threshold
        }
    """
    df = load_stream_source(shuffle=shuffle)
    if n is not None:
        df = df.head(n)

    feature_cols = config.ALL_FEATURE_COLUMNS
    true_labels = df[config.TARGET_COLUMN] if config.TARGET_COLUMN in df.columns else None
    features_only = df[feature_cols]

    for i in range(len(df)):
        row = features_only.iloc[[i]]
        scored = predict_batch(row)  # applies scaling + model prediction internally
        prob = float(scored["fraud_probability"].iloc[0])
        is_fraud_pred = bool(scored["is_fraud"].iloc[0])

        result = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "transaction": row.iloc[0].to_dict(),
            "true_label": int(true_labels.iloc[i]) if true_labels is not None else None,
            "fraud_probability": round(prob, 4),
            "is_fraud_predicted": is_fraud_pred,
            "alert": prob >= config.ALERT_PROBABILITY_THRESHOLD,
        }

        yield result

        if delay:
            time.sleep(delay)


def run_console_demo(n: int = 20, delay: float = 0.5):
    """Simple console demo of the live simulator (no Streamlit needed)."""
    logger.info(f"Starting console simulation for {n} transactions (delay={delay}s)...")
    correct = 0
    total = 0

    for result in stream_transactions(n=n, delay=delay):
        flag = "🚨 FRAUD ALERT" if result["alert"] else "  ok"
        print(
            f"[{result['timestamp']}] amount={result['transaction'].get(config.AMOUNT_COLUMN):.2f} "
            f"prob={result['fraud_probability']:.4f} {flag} "
            f"(true_label={result['true_label']})"
        )
        if result["true_label"] is not None:
            total += 1
            correct += int(result["is_fraud_predicted"] == bool(result["true_label"]))

    if total:
        logger.info(f"Console demo accuracy on streamed sample: {correct}/{total} = {correct/total:.2%}")


def main():
    parser = argparse.ArgumentParser(description="FraudShield AI live transaction simulator")
    parser.add_argument("--n", type=int, default=20, help="Number of transactions to simulate")
    parser.add_argument("--delay", type=float, default=config.SIMULATOR_STREAM_DELAY_SEC,
                         help="Delay in seconds between transactions")
    args = parser.parse_args()

    run_console_demo(n=args.n, delay=args.delay)


if __name__ == "__main__":
    main()
