import joblib
import pandas as pd

from src.config import PROCESSED_DIR, TRAIN_IDENTITY_FILE, TRAIN_TRANSACTION_FILE
from src.preprocessing.data_loader_merger import merge_transaction_identity
from src.preprocessing.encoder import CategoricalEncoder
from src.preprocessing.feature_selection import select_features
from src.preprocessing.missing_handler import MissingValueImputer, drop_extreme_missing
from src.preprocessing.temporal_features import add_temporal_features


def time_based_train_val_split(
    df: pd.DataFrame,
    val_size: float,
    time_col: str = "TransactionDT",
    transaction_id_col: str = "TransactionID",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if time_col not in df.columns:
        raise KeyError(f"Expected time_col='{time_col}' in dataframe columns.")
    if not 0 < val_size < 1:
        raise ValueError("val_size must be between 0 and 1.")
    if len(df) < 2:
        raise ValueError("Need at least two rows to create train and validation splits.")

    sort_cols = [time_col]
    if transaction_id_col in df.columns:
        sort_cols.append(transaction_id_col)

    df_sorted = df.sort_values(sort_cols, kind="mergesort").reset_index(drop=True)
    split_idx = max(1, min(len(df_sorted) - 1, int(len(df_sorted) * (1 - val_size))))

    train_df = df_sorted.iloc[:split_idx].copy()
    val_df = df_sorted.iloc[split_idx:].copy()
    return train_df, val_df


def run_full_pipeline(val_size: float = 0.2, save_artifacts: bool = True):
    # Merge transaction + identity data
    merged_df = merge_transaction_identity(
        TRAIN_TRANSACTION_FILE,
        TRAIN_IDENTITY_FILE,
        output_file=None,
    )

    # Split chronologically before any learned or history-based transforms.
    target = "isFraud"
    train_df, val_df = time_based_train_val_split(merged_df, val_size=val_size)

    print(f"Train: {len(train_df):,} rows (fraud rate: {train_df[target].mean():.4f})")
    print(f"Val  : {len(val_df):,} rows (fraud rate: {val_df[target].mean():.4f})")

    # Build temporal features on the combined chronological stream.
    # Validation rows only see earlier transactions, which matches deployment.
    combined_df = pd.concat([train_df, val_df], axis=0, ignore_index=True)
    combined_df = add_temporal_features(
        combined_df,
        time_col="TransactionDT",
        amount_col="TransactionAmt",
        entity_defs={
            "card1": ["card1"],
            "card1_addr1": ["card1", "addr1"],
            "card1_card2": ["card1", "card2"],
            "email_p": ["P_emaildomain"],
        },
        windows_seconds=[3600, 86400, 7 * 86400],
    )

    train_len = len(train_df)
    train_df = combined_df.iloc[:train_len].copy()
    val_df = combined_df.iloc[train_len:].copy()

    # Learn structural column drops on train only, then apply to validation.
    train_df, dropped_columns = drop_extreme_missing(train_df, threshold=90)
    val_df = val_df.drop(columns=[col for col in dropped_columns if col in val_df.columns])

    # Impute missing values fit on train, transform both.
    imputer = MissingValueImputer()
    train_df = imputer.fit_transform(train_df)
    val_df = imputer.transform(val_df)
    if save_artifacts:
        imputer.save()

    # Encode categorical columns fit on train, transform both.
    encoder = CategoricalEncoder()
    train_df = encoder.fit_transform(train_df, target_col=target)
    val_df = encoder.transform(val_df)
    if save_artifacts:
        encoder.save()

    # Feature selection fit on train, apply the same feature list to validation.
    train_df, selected_features = select_features(
        train_df,
        target_col=target,
        save_feature_list=save_artifacts,
    )
    val_df, _ = select_features(val_df, feature_list=selected_features)

    # Separate X and y for modeling.
    X_train = train_df.drop(columns=[target])
    y_train_out = train_df[target]
    X_val = val_df.drop(columns=[target])
    y_val_out = val_df[target]

    if "TransactionID" in X_train.columns:
        X_train = X_train.drop(columns=["TransactionID"])
    if "TransactionID" in X_val.columns:
        X_val = X_val.drop(columns=["TransactionID"])

    print(f"X_train: {X_train.shape} | X_val: {X_val.shape}")

    X_train = X_train.astype({col: "float32" for col in X_train.columns})
    X_val = X_val.astype({col: "float32" for col in X_val.columns})

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
