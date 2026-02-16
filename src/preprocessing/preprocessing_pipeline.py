from pathlib import Path
from src.config import *
from src.utils import *
from src.preprocessing.data_loader_merger import merge_transaction_identity
from src.preprocessing.temporal_features import add_temporal_features
from src.preprocessing.missing_handler import drop_extreme_missing, impute_missing_values
# from src.preprocessing.encoder import encode_categorical
# from src.preprocessing.feature_selection import select_features

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
    
    # Step 2: Temporal Feature Engineering
    df_temporal = add_temporal_features(
        merged_df,
        time_col="TransactionDT",
        amount_col="TransactionAmt",
        entity_defs={
            "card1": ["card1"],
            "card1_addr1": ["card1", "addr1"],
            "card1_card2": ["card1", "card2"],
            "email_p": ["P_emaildomain"],
        },
        windows_seconds=[3600, 86400, 7 * 86400], # 1 hour, 1 day, 1 week
    )
    
    # Step 3: Drop extreme missing columns (>90%)
    df_after_drop = drop_extreme_missing(
        df_temporal,
        threshold=90,
        output_file=PROCESSED_DIR / 'after_drop.csv' if save_intermediate else None
    )
    
    # Step 4: Impute remaining missing values
    df_imputed = impute_missing_values(
        df_after_drop,
        output_file=PROCESSED_DIR / 'after_impute.csv' if save_intermediate else None
    )
    return df_imputed

    # # Step 5: Encode categorical variables
    # df_encoded = encode_categorical(
    #     df_imputed,
    #     output_file=PROCESSED_DIR / 'after_encode.csv' if save_intermediate else None
    # )
    
    # # Step 6: Feature selection
    # df_final = select_features(
    #     df_encoded,
    #     method='correlation',
    #     output_file=FINAL_DATA_FILE  # Always save final result
    # )
    
    # return df_final