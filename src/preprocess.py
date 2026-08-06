"""
preprocess.py
-------------
Handles data loading, cleaning, feature scaling, train/test splitting,
and class-imbalance handling (SMOTE) for the FraudShield AI project.

Usage (CLI):
    python -m src.preprocess

This will read data/creditcard.csv, produce scaled/split data, and
save train.csv / test.csv into data/processed/, plus the fitted
scaler into models/scaler.pkl.
"""

import logging
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src import config

logging.basicConfig(level=config.LOG_LEVEL, format=config.LOG_FORMAT)
logger = logging.getLogger(__name__)


def load_raw_data(path=config.RAW_DATA_PATH) -> pd.DataFrame:
    """Load the raw credit card transactions CSV."""
    logger.info(f"Loading raw data from {path}")
    df = pd.read_csv(path)
    logger.info(f"Loaded data with shape {df.shape}")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Basic cleaning: drop duplicates and nulls."""
    before = len(df)
    df = df.drop_duplicates()
    df = df.dropna()
    after = len(df)
    logger.info(f"Cleaned data: removed {before - after} rows (duplicates/nulls)")
    return df.reset_index(drop=True)


def scale_features(df: pd.DataFrame, scaler: StandardScaler = None, fit: bool = True):
    """
    Scale 'Amount' and 'Time' columns (V1-V28 are already PCA-scaled
    in the standard Kaggle dataset). Returns the scaled dataframe and
    the fitted scaler.
    """
    df = df.copy()
    cols_to_scale = [config.AMOUNT_COLUMN, config.TIME_COLUMN]

    if scaler is None:
        scaler = StandardScaler()

    if fit:
        df[cols_to_scale] = scaler.fit_transform(df[cols_to_scale])
        logger.info("Fitted new scaler on Amount/Time columns")
    else:
        df[cols_to_scale] = scaler.transform(df[cols_to_scale])
        logger.info("Applied existing scaler to Amount/Time columns")

    return df, scaler


def split_data(df: pd.DataFrame):
    """Stratified train/test split to preserve fraud ratio in both sets."""
    X = df.drop(columns=[config.TARGET_COLUMN])
    y = df[config.TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE,
        stratify=y,
    )
    logger.info(
        f"Split data -> train: {X_train.shape}, test: {X_test.shape} "
        f"(train fraud rate: {y_train.mean():.5f}, test fraud rate: {y_test.mean():.5f})"
    )
    return X_train, X_test, y_train, y_test


def balance_classes(X_train, y_train):
    """Apply SMOTE oversampling to the training set only (never on test set)."""
    if not config.USE_SMOTE:
        return X_train, y_train

    from imblearn.over_sampling import SMOTE

    logger.info(
        f"Applying SMOTE with sampling_strategy={config.SMOTE_SAMPLING_STRATEGY}"
    )
    smote = SMOTE(
        sampling_strategy=config.SMOTE_SAMPLING_STRATEGY,
        random_state=config.RANDOM_STATE,
    )
    X_res, y_res = smote.fit_resample(X_train, y_train)
    logger.info(f"After SMOTE -> class counts: {pd.Series(y_res).value_counts().to_dict()}")
    return X_res, y_res


def run_pipeline(save: bool = True):
    """Full preprocessing pipeline: load -> clean -> scale -> split -> balance."""
    df = load_raw_data()
    df = clean_data(df)
    df, scaler = scale_features(df, fit=True)

    X_train, X_test, y_train, y_test = split_data(df)
    X_train_bal, y_train_bal = balance_classes(X_train, y_train)

    if save:
        joblib.dump(scaler, config.SCALER_PATH)
        train_out = X_train_bal.copy()
        train_out[config.TARGET_COLUMN] = y_train_bal
        test_out = X_test.copy()
        test_out[config.TARGET_COLUMN] = y_test

        train_out.to_csv(config.TRAIN_DATA_PATH, index=False)
        test_out.to_csv(config.TEST_DATA_PATH, index=False)
        logger.info(f"Saved processed train/test sets to {config.PROCESSED_DATA_DIR}")

    return X_train_bal, X_test, y_train_bal, y_test, scaler


if __name__ == "__main__":
    run_pipeline()
