"""
test_evaluate.py
-----------------
Unit tests for the metrics computation in src/evaluate.py, using
small hand-constructed prediction arrays with known correct answers.
"""

import numpy as np
import pytest

from src.evaluate import compute_metrics


def test_compute_metrics_perfect_classifier():
    y_true = np.array([0, 0, 0, 1, 1])
    y_pred = np.array([0, 0, 0, 1, 1])
    y_probs = np.array([0.05, 0.10, 0.20, 0.95, 0.99])

    metrics = compute_metrics(y_true, y_pred, y_probs)

    assert metrics["precision"] == pytest.approx(1.0)
    assert metrics["recall"] == pytest.approx(1.0)
    assert metrics["f1_score"] == pytest.approx(1.0)
    assert metrics["roc_auc"] == pytest.approx(1.0)
    assert metrics["pr_auc"] == pytest.approx(1.0)


def test_compute_metrics_contains_expected_keys():
    y_true = np.array([0, 1, 0, 1])
    y_pred = np.array([0, 0, 0, 1])
    y_probs = np.array([0.1, 0.4, 0.2, 0.8])

    metrics = compute_metrics(y_true, y_pred, y_probs)

    assert set(metrics.keys()) == {"precision", "recall", "f1_score", "roc_auc", "pr_auc"}


def test_compute_metrics_recall_drops_with_missed_fraud():
    y_true = np.array([1, 1, 1, 1])
    y_pred = np.array([1, 0, 0, 0])  # only caught 1 of 4 fraud cases
    y_probs = np.array([0.9, 0.3, 0.2, 0.1])

    metrics = compute_metrics(y_true, y_pred, y_probs)

    assert metrics["recall"] == pytest.approx(0.25)


def test_compute_metrics_precision_drops_with_false_alarms():
    y_true = np.array([0, 0, 0, 1])
    y_pred = np.array([1, 1, 0, 1])  # 2 false positives, 1 true positive
    y_probs = np.array([0.6, 0.55, 0.2, 0.9])

    metrics = compute_metrics(y_true, y_pred, y_probs)

    assert metrics["precision"] == pytest.approx(1 / 2)
