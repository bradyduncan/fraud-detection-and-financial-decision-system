import numpy as np
import pandas as pd
from pathlib import Path
import sys

from imblearn.over_sampling import SMOTE

# Allow running this file directly by adding repo root to sys.path
if __name__ == "__main__" and __package__ is None:
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

from src.config import RANDOM_SEED
from src.utils.data_splits import load_splits


def apply_smote(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    smote_sampling_strategy=0.2,
    smote_k_neighbors: int = 5,
    random_state: int = RANDOM_SEED,
):
    """
    Apply SMOTE to training data only.
    """

    print("Before SMOTE (train):")
    print(y_train.value_counts(normalize=True))

    # Ensure numeric features only
    if not np.all([np.issubdtype(dtype, np.number) for dtype in X_train.dtypes]):
        non_numeric = [c for c in X_train.columns if not np.issubdtype(X_train[c].dtype, np.number)]
        raise TypeError(
            "SMOTE requires numeric features. Non-numeric columns found: "
            f"{non_numeric[:20]}{'...' if len(non_numeric) > 20 else ''}"
        )

    minority_count = int((y_train == 1).sum())
    if minority_count <= 1:
        raise ValueError(
            f"Not enough minority samples to apply SMOTE. minority_count={minority_count}"
        )

    k = min(smote_k_neighbors, minority_count - 1)
    if k < 1:
        raise ValueError(
            f"Computed smote_k_neighbors={k}. Increase training size or adjust strategy."
        )

    smote = SMOTE(
        sampling_strategy=smote_sampling_strategy,
        k_neighbors=k,
        random_state=random_state,
    )

    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

    X_train_res = pd.DataFrame(X_train_res, columns=X_train.columns)
    y_train_res = pd.Series(y_train_res, name="isFraud")

    print("After SMOTE (train only):")
    print(y_train_res.value_counts(normalize=True))

    return X_train_res, y_train_res


if __name__ == "__main__":
    # 1) Load saved splits
    X_train, y_train, X_val, y_val = load_splits()

    # 2) Apply SMOTE to training set only
    X_train_res, y_train_res = apply_smote(X_train, y_train)

    # 3) X_val and y_val remain unchanged
    print("Validation shape:", X_val.shape)
