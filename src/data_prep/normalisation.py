"""
Feature normalization for fraud detection pipeline.
Fits on training data only to prevent data leakage.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
from src.config import *

SKIP_COLS = ["TransactionID", "isFraud"]


class FeatureNormalizer:
    def __init__(self):
        self.scaler = StandardScaler()
        self.feature_columns = []

    def fit(self, df):
        self.feature_columns = [c for c in df.columns if c not in SKIP_COLS]
        self.scaler.fit(df[self.feature_columns])
        return self

    def transform(self, df):
        df_norm = df.copy()
        df_norm[self.feature_columns] = self.scaler.transform(df[self.feature_columns])
        return df_norm

    def fit_transform(self, df):
        return self.fit(df).transform(df)

    def save(self, path=None):
        path = path or PROCESSED_DIR / "scaler.joblib"
        joblib.dump(self.scaler, path)
        print(f"Scaler saved to {path}")

    def load(self, path=None):
        path = path or PROCESSED_DIR / "scaler.joblib"
        self.scaler = joblib.load(path)
        print(f"Scaler loaded from {path}")
        return self


def normalize_data(train_df, test_df=None, output_dir=None):
    """
    Normalize features using StandardScaler.
    Fits on train only, transforms both train and test.

    Args:
        train_df: Training dataframe
        test_df: Test dataframe (optional)
        output_dir: Directory to save normalized CSVs

    Returns:
        train_norm: Normalized training data
        test_norm: Normalized test data (None if test_df not provided)
        normalizer: Fitted FeatureNormalizer object
    """
    normalizer = FeatureNormalizer()

    # Fit on train, transform train
    train_norm = normalizer.fit_transform(train_df)
    print(f"Normalized training data: {train_norm.shape}")

    # Transform test if provided
    test_norm = None
    if test_df is not None:
        test_norm = normalizer.transform(test_df)
        print(f"Normalized test data: {test_norm.shape}")

    # Save outputs
    if output_dir:
        output_dir = Path(output_dir)
        train_norm.to_csv(output_dir / "train_normalized.csv", index=False)
        if test_norm is not None:
            test_norm.to_csv(output_dir / "test_normalized.csv", index=False)
        normalizer.save(output_dir / "scaler.joblib")

    return train_norm, test_norm, normalizer