# Handle missing values in two phases:
# 1. Drop extreme missing columns (>90%)
# 2. Impute remaining missing values

import pandas as pd
import numpy as np
from pandas.api.types import (
    is_numeric_dtype,
    is_datetime64_any_dtype,
    is_categorical_dtype,
    is_bool_dtype,
    is_string_dtype,
    is_object_dtype,
)
from src.config import *
from src.utils import *

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

    print(f"\nDataset after removing Columns with more than 90% missing values:")
    print(f"Shape: {df_cleaned.shape}")
    # print(f"Columns after dropping extreme missing: {df_cleaned.columns.tolist()}")
    
    # Save if requested
    if output_file:
        df_cleaned.to_csv(output_file, index=False)
        print(f"Saved dataframe after dropping extreme missing columns to {output_file}")
    
    return df_cleaned


def impute_missing_values(df, output_file=None):
    df_imputed = df.copy()

    # 1. V features: Use -999 for high missing (>50%)
    v_features = [col for col in df_imputed.columns if col.startswith('V')]

    for col in v_features:
        if df_imputed[col].isnull().sum() > 0:
            missing_pct = df_imputed[col].isnull().sum() / len(df_imputed)

            if missing_pct > 0.50:
                # High missing: use special value
                df_imputed[col] = df_imputed[col].fillna(-999)
            else:
                # Low missing: use median
                median_val = df_imputed[col].median()
                df_imputed[col] = df_imputed[col].fillna(median_val)

    print(f"Imputed {len(v_features)} V features")

    # 2. Other columns: impute by dtype
    imputed_numeric = 0
    imputed_categorical = 0
    imputed_datetime = 0
    imputed_bool = 0

    for col in df_imputed.columns:
        if col in ["TransactionID", "isFraud"] or col.startswith("V"):
            continue

        missing = df_imputed[col].isnull().sum()
        if missing == 0:
            continue

        col_series = df_imputed[col]
        if is_numeric_dtype(col_series):
            median_val = col_series.median()
            if pd.isna(median_val):
                median_val = -999
            df_imputed[col] = col_series.fillna(median_val)
            imputed_numeric += 1
        elif is_datetime64_any_dtype(col_series):
            non_null = col_series.dropna()
            if non_null.empty:
                median_val = pd.Timestamp("1970-01-01")
            else:
                median_val = non_null.median()
            df_imputed[col] = col_series.fillna(median_val)
            imputed_datetime += 1
        elif is_bool_dtype(col_series):
            mode_val = col_series.dropna().mode()
            fill_val = bool(mode_val.iloc[0]) if not mode_val.empty else False
            df_imputed[col] = col_series.fillna(fill_val)
            imputed_bool += 1
        elif is_categorical_dtype(col_series) or is_string_dtype(col_series) or is_object_dtype(col_series):
            df_imputed[col] = col_series.fillna("Unknown")
            imputed_categorical += 1
        else:
            df_imputed[col] = col_series.fillna("Unknown")
            imputed_categorical += 1

    print(f"Imputed {imputed_numeric} numeric columns with median")
    print(f"Imputed {imputed_datetime} datetime columns with median")
    print(f"Imputed {imputed_bool} boolean columns with mode")
    print(f"Imputed {imputed_categorical} categorical/string columns with 'Unknown'")

    # Verification: Check no missing values remain
    remaining_missing = df_imputed.isnull().sum().sum()
    if remaining_missing != 0:
        missing_by_col = df_imputed.isnull().sum()
        top_missing = missing_by_col[missing_by_col > 0].sort_values(ascending=False).head(10)
        raise AssertionError(
            f"Still have {remaining_missing} missing values! Top missing columns:\n{top_missing}"
        )
    print("Zero missing values remaining")

    # Save if requested
    if output_file:
        df_imputed.to_csv(output_file, index=False)

    return df_imputed
def main():
    df = pd.read_csv(MERGED_DATA_FILE)
    
    # Test drop
    df_dropped = drop_extreme_missing(df, threshold=90)
     
    # # Test impute
    df_imputed = impute_missing_values(df_dropped)
    
    return df_imputed


if __name__ == "__main__":
    main()

