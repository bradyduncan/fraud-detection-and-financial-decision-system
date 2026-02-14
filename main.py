from pathlib import Path
from src.config import *
from src.utils import *
from src.data_loader_merger import merge_transaction_identity
from src.missing_handler import drop_extreme_missing, impute_missing_values
# from src.encoder import encode_categorical
# from src.feature_selection import select_features

def run_full_pipeline(save_intermediate=False):
    """
    Run preprocessing pipeline in correct order:
    1. Merge transaction + identity
    2. Drop columns with >90% missing
    3. Impute remaining missing values
    4. Encode categorical variables
    5. Feature selection
    
    Args:
        save_intermediate: Save CSV at each step (for debugging)
    
    Returns:
        final_df: Preprocessed dataframe ready for modeling
    """
    
    # Step 1: Merge data
    merged_df = merge_transaction_identity(
        TRAIN_TRANSACTION_FILE,
        TRAIN_IDENTITY_FILE,
        output_file=MERGED_DATA_FILE if save_intermediate else None
    )
    
    # Step 2: Drop extreme missing columns (>90%)
    df_after_drop = drop_extreme_missing(
        merged_df,
        threshold=90,
        output_file=PROCESSED_DIR / 'after_drop.csv' if save_intermediate else None
    )
    
    # Step 3: Impute remaining missing values
    df_imputed = impute_missing_values(
        df_after_drop,
        output_file=PROCESSED_DIR / 'after_impute.csv' if save_intermediate else None
    )
    return df_imputed

    # # Step 4: Encode categorical variables
    # df_encoded = encode_categorical(
    #     df_imputed,
    #     output_file=PROCESSED_DIR / 'after_encode.csv' if save_intermediate else None
    # )
    
    # # Step 5: Feature selection
    # df_final = select_features(
    #     df_encoded,
    #     method='correlation',
    #     output_file=FINAL_DATA_FILE  # Always save final result
    # )
    
    # return df_final


if __name__ == "__main__":
    # Run pipeline
    df = run_full_pipeline(save_intermediate=False)