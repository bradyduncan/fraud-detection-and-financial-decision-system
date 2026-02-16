# Handle missing values in two phases:
# 1. Drop extreme missing columns (>90%)
# 2. Impute remaining missing values

import pandas as pd
import numpy as np
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
                df_imputed[col] = df_imputed[col].fillna(-999)  # ✅ Fixed
            else:
                # Low missing: use median
                median_val = df_imputed[col].median()
                df_imputed[col] = df_imputed[col].fillna(median_val)  # ✅ Fixed
    
    print(f"Imputed {len(v_features)} V features")
    
    # 2. Other numeric features: Use median
    numeric_cols = df_imputed.select_dtypes(include=[np.number]).columns
    numeric_cols = [col for col in numeric_cols 
                   if col not in ['TransactionID', 'isFraud'] 
                   and not col.startswith('V')]
    
    imputed_count = 0
    for col in numeric_cols:
        if df_imputed[col].isnull().sum() > 0:
            median_val = df_imputed[col].median()
            df_imputed[col] = df_imputed[col].fillna(median_val)  # ✅ Fixed
            imputed_count += 1
    
    print(f"Imputed {imputed_count} numeric columns with median")
    
    # 3. Categorical features: Use "Unknown"
    categorical_cols = df_imputed.select_dtypes(include=['object']).columns
    
    imputed_count = 0
    for col in categorical_cols:
        if df_imputed[col].isnull().sum() > 0:
            df_imputed[col] = df_imputed[col].fillna('Unknown')  # ✅ Fixed
            imputed_count += 1
    
    print(f"Imputed {imputed_count} categorical columns with 'Unknown'")
    
    # Verification: Check no missing values remain
    remaining_missing = df_imputed.isnull().sum().sum()
    assert remaining_missing == 0, f"Still have {remaining_missing} missing values!"
    print(f"Zero missing values remaining")
    
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