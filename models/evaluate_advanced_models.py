"""
Evaluate saved advanced fraud-detection models.

Run from the PROJECT ROOT:
    python src/evaluate_advanced_models.py
"""

from pathlib import Path
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    matthews_corrcoef,
    confusion_matrix,
)

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "processed" / "processed_creditcard.csv"
MODEL_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"

def evaluate(name, model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

    return {
        "model": name,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "pr_auc": average_precision_score(y_test, y_prob),
        "roc_auc": roc_auc_score(y_test, y_prob),
        "mcc": matthews_corrcoef(y_test, y_pred),
        "false_negative_rate": fn / (fn + tp) if (fn + tp) else 0.0,
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }

def main():
    df = pd.read_csv(DATA_PATH)
    X = df.drop(columns=["Class"])
    y = df["Class"]

    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    results = []
    for name in ["xgboost", "lightgbm", "catboost", "advanced_best_model"]:
        path = MODEL_DIR / f"{name}.pkl"
        if path.exists():
            model = joblib.load(path)
            results.append(evaluate(name, model, X_test, y_test))

    results_df = pd.DataFrame(results).sort_values("f1", ascending=False)
    RESULTS_DIR.mkdir(exist_ok=True)
    results_df.to_csv(RESULTS_DIR / "advanced_model_evaluation.csv", index=False)
    print(results_df.to_string(index=False))

if __name__ == "__main__":
    main()
