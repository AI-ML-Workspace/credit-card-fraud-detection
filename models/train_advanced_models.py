"""
Train XGBoost, LightGBM and CatBoost for the Credit Card Fraud Detection project.

Expected project structure:
credit-card-fraud-detection/
├── data/
│   └── processed/
│       └── processed_creditcard.csv
├── models/
├── results/
└── src/
    └── train_advanced_models.py

Run from the PROJECT ROOT:
    python src/train_advanced_models.py
"""

from pathlib import Path
import json
import warnings

import joblib
import numpy as np
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

warnings.filterwarnings("ignore")

# Optional third-party libraries required for this script.
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "processed" / "processed_creditcard.csv"
MODEL_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"

MODEL_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)


def evaluate_model(name, model, X_test, y_test):
    """Evaluate a binary fraud classifier using imbalance-aware metrics."""
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "model": name,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "pr_auc": average_precision_score(y_test, y_proba),
        "roc_auc": roc_auc_score(y_test, y_proba),
        "mcc": matthews_corrcoef(y_test, y_pred),
        "tn": int(confusion_matrix(y_test, y_pred)[0, 0]),
        "fp": int(confusion_matrix(y_test, y_pred)[0, 1]),
        "fn": int(confusion_matrix(y_test, y_pred)[1, 0]),
        "tp": int(confusion_matrix(y_test, y_pred)[1, 1]),
    }
    return metrics


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Processed dataset not found at: {DATA_PATH}\n"
            "Place processed_creditcard.csv in data/processed/."
        )

    df = pd.read_csv(DATA_PATH)

    if "Class" not in df.columns:
        raise ValueError("Target column 'Class' was not found.")

    X = df.drop(columns=["Class"])
    y = df["Class"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    # Fraud is extremely rare in this dataset. Use the training split only
    # to calculate the imbalance ratio.
    negative = int((y_train == 0).sum())
    positive = int((y_train == 1).sum())
    scale_pos_weight = negative / positive

    print(f"Training shape: {X_train.shape}")
    print(f"Testing shape:  {X_test.shape}")
    print(f"Fraud cases in training set: {positive}")
    print(f"scale_pos_weight: {scale_pos_weight:.4f}")

    models = {
        "xgboost": XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="binary:logistic",
            eval_metric="logloss",
            scale_pos_weight=scale_pos_weight,
            random_state=42,
            n_jobs=-1,
            tree_method="hist",
        ),
        "lightgbm": LGBMClassifier(
            n_estimators=300,
            max_depth=-1,
            num_leaves=31,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            class_weight="balanced",
            objective="binary",
            random_state=42,
            n_jobs=-1,
            verbosity=-1,
        ),
        "catboost": CatBoostClassifier(
            iterations=300,
            depth=6,
            learning_rate=0.05,
            loss_function="Logloss",
            eval_metric="AUC",
            auto_class_weights="Balanced",
            random_seed=42,
            verbose=False,
            thread_count=-1,
        ),
    }

    all_results = []

    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)

        model_path = MODEL_DIR / f"{name}.pkl"
        joblib.dump(model, model_path)

        result = evaluate_model(name, model, X_test, y_test)
        all_results.append(result)

        print(
            f"{name}: "
            f"Accuracy={result['accuracy']:.6f}, "
            f"Precision={result['precision']:.6f}, "
            f"Recall={result['recall']:.6f}, "
            f"F1={result['f1']:.6f}, "
            f"PR-AUC={result['pr_auc']:.6f}, "
            f"ROC-AUC={result['roc_auc']:.6f}, "
            f"MCC={result['mcc']:.6f}"
        )
        print(f"Saved: {model_path}")

    results_df = pd.DataFrame(all_results).sort_values("f1", ascending=False)
    results_df.to_csv(RESULTS_DIR / "advanced_model_results.csv", index=False)

    with open(RESULTS_DIR / "advanced_model_results.json", "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)

    # Save the strongest model by F1 for downstream deployment.
    best_name = results_df.iloc[0]["model"]
    best_model = models[best_name]
    best_path = MODEL_DIR / "advanced_best_model.pkl"
    joblib.dump(best_model, best_path)

    with open(RESULTS_DIR / "advanced_best_model.txt", "w", encoding="utf-8") as f:
        f.write(best_name)

    print("\n=== FINAL ADVANCED MODEL SELECTION ===")
    print(results_df[["model", "accuracy", "precision", "recall", "f1", "pr_auc", "roc_auc", "mcc"]])
    print(f"\nBest model by F1: {best_name}")
    print(f"Saved to: {best_path}")


if __name__ == "__main__":
    main()
