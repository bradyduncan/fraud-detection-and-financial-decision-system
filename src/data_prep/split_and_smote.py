import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE


def split_and_smote(
    train_final: pd.DataFrame,
    target_col: str = "isFraud",
    test_size: float = 0.2,
    random_state: int = 42,
    smote_sampling_strategy=0.2,
    smote_k_neighbors: int = 5,
):
    """
    Split train_final into train/val, then apply SMOTE to the train split only.

    Args:
        train_final: Preprocessed dataframe containing features + target.
        target_col: Name of the target column.
        test_size: Validation fraction.
        random_state: Reproducibility.
        smote_sampling_strategy: 'auto' or float or dict to determine SMOTE resampling.
        smote_k_neighbors: Neighbors used by SMOTE.

    Returns:
        X_train_res, y_train_res, X_val, y_val
    """
    print("Before SMOTE:", train_final["isFraud"].value_counts(normalize=True))
    
    if target_col not in train_final.columns:
        raise ValueError(f"target_col='{target_col}' not found in train_final columns.")

    # Create copy
    df = train_final.copy()

    # Separate X/y
    y = df[target_col].astype(int)
    X = df.drop(columns=[target_col])

    # Check for numeric features only
    if not np.all([np.issubdtype(dtype, np.number) for dtype in X.dtypes]):
        non_numeric = [c for c in X.columns if not np.issubdtype(X[c].dtype, np.number)]
        raise TypeError(
            "SMOTE requires numeric features. Non-numeric columns found: "
            f"{non_numeric[:20]}{'...' if len(non_numeric) > 20 else ''}"
        )

    # Split first
    X_train, X_val, y_train, y_val = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    # If minority class is extremely small, k_neighbors must be < minority_count
    minority_count = int((y_train == 1).sum())
    if minority_count <= 1:
        raise ValueError(
            f"Not enough minority samples in training split to apply SMOTE. "
            f"minority_count={minority_count}"
        )
    k = min(smote_k_neighbors, minority_count - 1)
    if k < 1:
        raise ValueError(
            f"Computed smote_k_neighbors={k}. Increase training size or use different strategy."
        )

    smote = SMOTE(
        sampling_strategy=smote_sampling_strategy,
        k_neighbors=k,
        random_state=random_state,
    )

    # Transform only training data
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    X_train_res = pd.DataFrame(X_train_res, columns=X_train.columns)
    y_train_res = pd.Series(y_train_res, name=target_col)
    
    print("After SMOTE (train only):", y_train_res.value_counts(normalize=True))

    return X_train_res, y_train_res, X_val, y_val


if __name__ == "__main__":
    df = pd.read_csv("data/processed/train_final.csv")
    X_train_res, y_train_res, X_val, y_val = split_and_smote(df)
