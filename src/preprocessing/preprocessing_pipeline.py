from sklearn.model_selection import train_test_split
from src.preprocessing.data_loader_merger import merge_transaction_identity
from src.preprocessing.temporal_features import add_temporal_features
from src.preprocessing.missing_handler import drop_extreme_missing, MissingValueImputer
from src.preprocessing.encoder import CategoricalEncoder
from src.preprocessing.feature_selection import select_features
import joblib
from src.config import (
    TRAIN_TRANSACTION_FILE, 
    TRAIN_IDENTITY_FILE, 
    RANDOM_SEED, 
    PROCESSED_DIR
)


def run_full_pipeline(val_size: float = 0.2, save_artifacts: bool = True):
    # Step 1: Merge transaction + identity data
    merged_df = merge_transaction_identity(
        TRAIN_TRANSACTION_FILE,
        TRAIN_IDENTITY_FILE,
        output_file=None,
    )

    # Step 2: Temporal feature engineering
    # Temporarily disabled for testing — skip expensive temporal feature generation
    # To re-enable, restore the call to `add_temporal_features(...)` below.
    df_temporal = add_temporal_features(
        merged_df,
        time_col="TransactionDT",
        amount_col="TransactionAmt",
        entity_defs={
            "card1":       ["card1"],
            "card1_addr1": ["card1", "addr1"],
            "card1_card2": ["card1", "card2"],
            "email_p":     ["P_emaildomain"],
        },
        windows_seconds=[3600, 86400, 7 * 86400],
    )

    # Step 3: Drop columns with >90% missing
    # Structural filter — no statistics learned, safe to apply before split
    df_after_drop, _ = drop_extreme_missing(df_temporal, threshold=90)

    # Step 4: Split into train and validation sets
    target = "isFraud"
    y_full = df_after_drop[target].astype(int)
    X_full = df_after_drop.drop(columns=[target, "TransactionID"])

    X_train_raw, X_val_raw, y_train, y_val = train_test_split(
        X_full,
        y_full,
        test_size=val_size,
        stratify=y_full,
        random_state=RANDOM_SEED,
    )

    print(f"Train: {len(X_train_raw):,} rows (fraud rate: {y_train.mean():.4f})")
    print(f"Val  : {len(X_val_raw):,} rows (fraud rate: {y_val.mean():.4f})")

    # Reconstruct full row dataframes (features + target) for preprocessing steps
    train_df = X_train_raw.copy()
    train_df[target] = y_train.values
    val_df = X_val_raw.copy()
    val_df[target] = y_val.values

    # Step 5: Impute missing values — fit on train, transform both
    imputer = MissingValueImputer()
    train_df = imputer.fit_transform(train_df)
    val_df   = imputer.transform(val_df)
    if save_artifacts:
        imputer.save()

    # Step 6: Encode categorical columns — fit on train, transform both
    encoder = CategoricalEncoder()
    train_df = encoder.fit_transform(train_df, target_col=target)
    val_df   = encoder.transform(val_df)
    if save_artifacts:
        encoder.save()

    # Step 7: Feature selection — fit on train, apply same feature list to val
    train_df, selected_features = select_features(
        train_df,
        target_col=target,
        save_feature_list=save_artifacts,
    )
    val_df, _ = select_features(val_df, feature_list=selected_features)

    # Separate X and y for modeling
    X_train     = train_df.drop(columns=[target])
    y_train_out = train_df[target]
    X_val       = val_df.drop(columns=[target])
    y_val_out   = val_df[target]

    print(f"X_train: {X_train.shape} | X_val: {X_val.shape}")

    X_train = X_train.astype({col: "float32" for col in X_train.columns})
    X_val   = X_val.astype({col: "float32" for col in X_val.columns})
    
    if save_artifacts:
        splits_dir = PROCESSED_DIR / "splits"
        splits_dir.mkdir(parents=True, exist_ok=True)
        
        joblib.dump(X_train, splits_dir / "X_train.joblib")
        joblib.dump(y_train_out, splits_dir / "y_train.joblib")
        joblib.dump(X_val, splits_dir / "X_val.joblib")
        joblib.dump(y_val_out, splits_dir / "y_val.joblib")
        
        print(f"Splits saved to {splits_dir}")

    return X_train, y_train_out, X_val, y_val_out


if __name__ == "__main__":
    X_train, y_train, X_val, y_val = run_full_pipeline()