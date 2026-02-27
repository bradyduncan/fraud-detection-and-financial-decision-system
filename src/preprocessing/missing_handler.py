# Handle missing values in two phases:
# 1. Drop extreme missing columns (>90%)
# 2. Impute remaining missing values

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from pandas.api.types import (
    is_numeric_dtype,
    is_datetime64_any_dtype,
    is_bool_dtype,
    is_string_dtype,
    is_object_dtype,
)
from src.config import *

# Phase 1: Drop columns with extreme missingness
def drop_extreme_missing(df, threshold=90, output_file=None):
    df_cleaned = df.copy()
    
    # Calculate missing percentages
    missing_pct = (df_cleaned.isnull().sum() / len(df_cleaned)) * 100
    
    # Find columns to drop
    columns_to_drop = missing_pct[missing_pct > threshold].index.tolist()
    
    # Protect critical features
    columns_to_drop = [col for col in columns_to_drop if col not in CRITICAL_FEATURES]
    
    # Drop columns
    df_cleaned = df_cleaned.drop(columns=columns_to_drop)

    print(f"\nDataset after removing Columns with more than {threshold}% missing values:")
    print(f"  Dropped {len(columns_to_drop)} columns")
    print(f"  Shape: {df_cleaned.shape}")
    
    # Save if requested
    if output_file:
        df_cleaned.to_csv(output_file, index=False)
        print(f"  Saved to {output_file}")
    
    return df_cleaned, columns_to_drop


# Phase 2: MissingValueImputer — fit on TRAIN only, transform any split
class MissingValueImputer:
    """
    Learns imputation fill-values from training data only.
    Then applies the same fill-values to validation/test data.

    Imputation strategy:
      - V features with >50% missing  → -999 (sentinel value)
      - V features with <=50% missing → median
      - Numeric columns               → median
      - Bool columns                  → mode
      - Datetime columns              → median
      - Categorical / string / object → "Unknown"
    """

    def __init__(self):
        self.fill_values: dict = {}   # col → fill value
        self._fitted: bool = False

    def fit(self, df: pd.DataFrame) -> "MissingValueImputer": 
        # Learn fill-values from df (should be the TRAINING set only).
        self.fill_values = {}

        # --- V features ---
        v_features = [col for col in df.columns if col.startswith("V")]
        for col in v_features:
            if df[col].isnull().sum() == 0:
                continue
            missing_pct = df[col].isnull().sum() / len(df)
            if missing_pct > 0.50:
                self.fill_values[col] = -999
            else:
                self.fill_values[col] = df[col].median()

        # --- All other columns ---
        for col in df.columns:
            if col in ["TransactionID", "isFraud"] or col.startswith("V"):
                continue
            if df[col].isnull().sum() == 0:
                continue

            col_series = df[col]
            if is_numeric_dtype(col_series):
                median_val = col_series.median()
                self.fill_values[col] = median_val if not pd.isna(median_val) else -999
            elif is_datetime64_any_dtype(col_series):
                non_null = col_series.dropna()
                self.fill_values[col] = (
                    non_null.median() if not non_null.empty else pd.Timestamp("1970-01-01")
                )
            elif is_bool_dtype(col_series):
                mode_val = col_series.dropna().mode()
                self.fill_values[col] = bool(mode_val.iloc[0]) if not mode_val.empty else False
            elif isinstance(col_series.dtype, pd.CategoricalDtype) or is_string_dtype(col_series) or is_object_dtype(col_series):
                self.fill_values[col] = "Unknown"
            else:
                self.fill_values[col] = "Unknown"

        self._fitted = True
        print(f"[MissingValueImputer] Fitted fill-values for {len(self.fill_values)} columns")
        return self

    def transform(self, df: pd.DataFrame, output_file=None) -> pd.DataFrame:
        # Apply learned fill-values to df (train or val/test).
    
        if not self._fitted:
            raise RuntimeError("Call .fit() before .transform()")

        df_imputed = df.copy()

        for col, fill_val in self.fill_values.items():
            if col not in df_imputed.columns:
                continue
            df_imputed[col] = df_imputed[col].fillna(fill_val)

        # Verify
        remaining = df_imputed.isnull().sum().sum()
        if remaining != 0:
            missing_by_col = df_imputed.isnull().sum()
            top_missing = missing_by_col[missing_by_col > 0].sort_values(ascending=False).head(10)
            raise AssertionError(
                f"Still have {remaining} missing values after imputation!\n{top_missing}"
            )
        print(f"[MissingValueImputer] Zero missing values remaining after transform")

        if output_file:
            df_imputed.to_csv(output_file, index=False)

        return df_imputed

    def fit_transform(self, df: pd.DataFrame, output_file=None) -> pd.DataFrame:
        return self.fit(df).transform(df, output_file=output_file)

    def save(self, path=None):
        path = Path(path or PROCESSED_DIR / "models" / "imputer.joblib")
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        print(f"[MissingValueImputer] Saved to {path}")
        return path

    @classmethod
    def load(cls, path=None) -> "MissingValueImputer":
        path = Path(path or PROCESSED_DIR / "models" / "imputer.joblib")
        obj = joblib.load(path)
        print(f"[MissingValueImputer] Loaded from {path}")
        return obj

# Kept for backwards compatibility
def impute_missing_values(df, output_file=None):
    """Legacy function — fits AND transforms on the same df (use MissingValueImputer for proper train/val workflow)."""
    imputer = MissingValueImputer()
    return imputer.fit_transform(df, output_file=output_file)


def main():
    df = pd.read_csv(MERGED_DATA_FILE)
    df_dropped, _ = drop_extreme_missing(df, threshold=90)
    df_imputed = impute_missing_values(df_dropped)
    return df_imputed


if __name__ == "__main__":
    main()
