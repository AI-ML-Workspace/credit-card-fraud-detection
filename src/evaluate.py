"""
evaluate.py
-----------
Evaluates the final saved model on the held-out test set: computes
precision, recall, F1, ROC-AUC, PR-AUC, confusion matrix, and saves
plots + a JSON metrics report to reports/.

Usage (CLI):
    python -m src.evaluate
"""

import json
import logging
import joblib
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # headless backend, safe for servers/CI
import matplotlib.pyplot as plt
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score,
    average_precision_score, confusion_matrix, roc_curve,
    classification_report,
)

from src import config

logging.basicConfig(level=config.LOG_LEVEL, format=config.LOG_FORMAT)
logger = logging.getLogger(__name__)


def load_test_data():
    test_df = pd.read_csv(config.TEST_DATA_PATH)
    X_test = test_df.drop(columns=[config.TARGET_COLUMN])
    y_test = test_df[config.TARGET_COLUMN]
    return X_test, y_test


def load_final_model():
    return joblib.load(config.FINAL_MODEL_PATH)


def compute_metrics(y_true, y_pred, y_probs) -> dict:
    metrics = {
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1_score": f1_score(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, y_probs),
        "pr_auc": average_precision_score(y_true, y_probs),
    }
    logger.info(f"Metrics: {json.dumps(metrics, indent=2)}")
    return metrics


def plot_confusion_matrix(y_true, y_pred, save_path=config.CONFUSION_MATRIX_PATH):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Legit", "Fraud"])
    ax.set_yticklabels(["Legit", "Fraud"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                     color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.colorbar(im)
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    logger.info(f"Saved confusion matrix plot to {save_path}")


def plot_roc(y_true, y_probs, save_path=config.ROC_CURVE_PATH):
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    auc = roc_auc_score(y_true, y_probs)

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(fpr, tpr, label=f"ROC curve (AUC = {auc:.4f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random guess")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    logger.info(f"Saved ROC curve plot to {save_path}")


def save_metrics_report(metrics: dict, y_true, y_pred, path=config.METRICS_REPORT_PATH):
    report = {
        "metrics": metrics,
        "classification_report": classification_report(y_true, y_pred, output_dict=True),
    }
    with open(path, "w") as f:
        json.dump(report, f, indent=2)
    logger.info(f"Saved evaluation report to {path}")


def run_evaluation():
    model = load_final_model()
    X_test, y_test = load_test_data()

    y_probs = model.predict_proba(X_test)[:, 1]
    y_pred = (y_probs >= config.DECISION_THRESHOLD).astype(int)

    metrics = compute_metrics(y_test, y_pred, y_probs)
    plot_confusion_matrix(y_test, y_pred)
    plot_roc(y_test, y_probs)
    save_metrics_report(metrics, y_test, y_pred)

    return metrics


if __name__ == "__main__":
    run_evaluation()
