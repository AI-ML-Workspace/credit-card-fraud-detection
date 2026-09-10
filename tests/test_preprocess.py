"""
test_preprocess.py
-------------------
Unit tests for src/preprocess.py, using small synthetic dataframes
that match the real dataset's schema (Time, V1-V28, Amount, Class)
so they run without needing creditcard.csv.
"""

import numpy as np
import pandas as pd
import pytest

from src import config
from src.preprocess import balance_classes, clean_data, scale_features, split_data


def make_synthetic_df(n=40, n_fraud=6, seed=0):
    rng = np.random.default_rng(seed)
    data = {config.TIME_COLUMN: rng.integers(0, 100_000, size=n).astype(float)}
    for col in config.FEATURE_COLUMNS_V:
        data[col] = rng.normal(size=n)
    data[config.AMOUNT_COLUMN] = rng.uniform(1, 500, size=n)

    labels = np.zeros(n, dtype=int)
    labels[:n_fraud] = 1
    rng.shuffle(labels)
    data[config.TARGET_COLUMN] = labels

    return pd.DataFrame(data)


def test_clean_data_removes_duplicates_and_nulls():
    df = make_synthetic_df(n=20, n_fraud=4)
    df = pd.concat([df, df.iloc[[0, 1]]], ignore_index=True)  # inject duplicates
    df.loc[2, config.AMOUNT_COLUMN] = np.nan  # inject a null

    cleaned = clean_data(df)

    assert cleaned.duplicated().sum() == 0
    assert cleaned.isna().sum().sum() == 0
    assert len(cleaned) == len(df.drop_duplicates().dropna())


def test_scale_features_centers_amount_and_time():
    df = make_synthetic_df(n=50, n_fraud=8)

    scaled_df, _scaler = scale_features(df, fit=True)

    for col in (config.AMOUNT_COLUMN, config.TIME_COLUMN):
        assert abs(scaled_df[col].mean()) < 1e-6

    # V-columns should be passed through untouched
    pd.testing.assert_series_equal(
        scaled_df[config.FEATURE_COLUMNS_V[0]], df[config.FEATURE_COLUMNS_V[0]]
    )


def test_scale_features_reuses_fitted_scaler_without_refitting():
    df = make_synthetic_df(n=50, n_fraud=8)
    _, scaler = scale_features(df, fit=True)

    new_df = make_synthetic_df(n=10, n_fraud=2, seed=1)
    _transformed, same_scaler = scale_features(new_df, scaler=scaler, fit=False)

    assert same_scaler is scaler


def test_split_data_is_stratified(monkeypatch):
    df = make_synthetic_df(n=100, n_fraud=20)
    monkeypatch.setattr(config, "TEST_SIZE", 0.25)

    X_train, X_test, y_train, y_test = split_data(df)

    assert len(X_test) == pytest.approx(len(df) * 0.25, abs=1)
    assert abs(y_train.mean() - y_test.mean()) < 0.05


def test_balance_classes_noop_when_smote_disabled(monkeypatch):
    df = make_synthetic_df(n=100, n_fraud=10)
    X_train, X_test, y_train, y_test = split_data(df)

    monkeypatch.setattr(config, "USE_SMOTE", False)
    X_bal, y_bal = balance_classes(X_train, y_train)

    assert len(X_bal) == len(X_train)
    pd.testing.assert_series_equal(y_bal.reset_index(drop=True), y_train.reset_index(drop=True))


def test_balance_classes_applies_smote_when_enabled(monkeypatch):
    pytest.importorskip("imblearn")
    df = make_synthetic_df(n=200, n_fraud=20)
    X_train, X_test, y_train, y_test = split_data(df)

    monkeypatch.setattr(config, "USE_SMOTE", True)
    monkeypatch.setattr(config, "SMOTE_SAMPLING_STRATEGY", 0.5)

    X_bal, y_bal = balance_classes(X_train, y_train)

    assert len(X_bal) > len(X_train)
    assert pd.Series(y_bal).mean() > y_train.mean()
