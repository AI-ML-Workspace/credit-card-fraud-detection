"""
test_predict.py
----------------
Unit tests for src/predict.py. Monkeypatches the model/scaler loaders
with deterministic dummies so the tests don't depend on a real trained
final_model.pkl / scaler.pkl being present.
"""

import numpy as np
import pandas as pd
import pytest

from src import config, predict


class DummyScaler:
    """Identity scaler — keeps the test's expected outputs simple/deterministic."""

    def transform(self, X):
        return X


class DummyModel:
    """Flags a transaction as fraud (prob=0.9) if Amount > 1000, else legit (prob=0.1)."""

    def predict_proba(self, X):
        amounts = X[config.AMOUNT_COLUMN].to_numpy()
        fraud_prob = np.where(amounts > 1000, 0.9, 0.1)
        return np.column_stack([1 - fraud_prob, fraud_prob])


@pytest.fixture(autouse=True)
def patch_model_and_scaler(monkeypatch):
    predict._model = None
    predict._scaler = None
    monkeypatch.setattr(predict, "_get_model", lambda: DummyModel())
    monkeypatch.setattr(predict, "_get_scaler", lambda: DummyScaler())
    yield
    predict._model = None
    predict._scaler = None


def _sample_transaction(amount):
    row = {config.TIME_COLUMN: 5000.0, config.AMOUNT_COLUMN: amount}
    row.update({col: 0.0 for col in config.FEATURE_COLUMNS_V})
    return row


def test_predict_transaction_flags_high_amount_as_fraud():
    result = predict.predict_transaction(_sample_transaction(2500))

    assert result["is_fraud"] is True
    assert result["fraud_probability"] == pytest.approx(0.9)


def test_predict_transaction_flags_low_amount_as_legit():
    result = predict.predict_transaction(_sample_transaction(50))

    assert result["is_fraud"] is False
    assert result["fraud_probability"] == pytest.approx(0.1)


def test_predict_batch_adds_expected_columns_and_values():
    df = pd.DataFrame([_sample_transaction(2500), _sample_transaction(50)])

    result = predict.predict_batch(df)

    assert "fraud_probability" in result.columns
    assert "is_fraud" in result.columns
    assert result["is_fraud"].tolist() == [True, False]


def test_preprocess_for_inference_preserves_feature_order():
    df = pd.DataFrame([_sample_transaction(200)])

    processed = predict.preprocess_for_inference(df)

    assert list(processed.columns) == config.ALL_FEATURE_COLUMNS
