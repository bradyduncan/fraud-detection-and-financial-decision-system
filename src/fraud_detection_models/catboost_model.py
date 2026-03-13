import sys
from pathlib import Path

import joblib
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

if __name__ == "__main__" and __package__ is None:
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

from src.config import PROCESSED_DIR, RANDOM_SEED
from src.fraud_detection_models.experimentation import test_scale_pos_weights
from src.utils.data_splits import load_splits

__all__ = [
    "train_and_evaluate",
    "save_model",
    "load_splits",
    "test_scale_pos_weights",
]


def train_and_evaluate(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    threshold: float = 0.5,
    model_params: dict | None = None,
):
    """
    Train CatBoost on train set and evaluate on validation set.
    """
    params = {
        "iterations": 4000,
        "depth": 6,
        "learning_rate": 0.1,
        "loss_function": "Logloss",
        "eval_metric": "AUC",
        "random_seed": RANDOM_SEED,
        "verbose": False,
        "allow_writing_files": False,
        "od_type": "Iter",
        "od_wait": 20,
        "l2_leaf_reg": 5,
        "subsample": 0.8,
    }
    if "scale_pos_weight" not in params:
        pos = int((y_train == 1).sum())
        neg = int((y_train == 0).sum())
        if pos > 0:
            params["scale_pos_weight"] = 0.65 * neg / pos
    if model_params:
        params.update(model_params)

    model = CatBoostClassifier(**params)
    model.fit(X_train, y_train, eval_set=(X_val, y_val), use_best_model=True)

    train_proba = model.predict_proba(X_train)[:, 1]
    train_pred = (train_proba >= threshold).astype(int)
    val_proba = model.predict_proba(X_val)[:, 1]
    val_pred = (val_proba >= threshold).astype(int)

    train_metrics = {
        "roc_auc": roc_auc_score(y_train, train_proba),
        "accuracy": accuracy_score(y_train, train_pred),
        "precision": precision_score(y_train, train_pred, zero_division=0),
        "recall": recall_score(y_train, train_pred, zero_division=0),
        "f1": f1_score(y_train, train_pred, zero_division=0),
    }
    val_metrics = {
        "roc_auc": roc_auc_score(y_val, val_proba),
        "accuracy": accuracy_score(y_val, val_pred),
        "precision": precision_score(y_val, val_pred, zero_division=0),
        "recall": recall_score(y_val, val_pred, zero_division=0),
        "f1": f1_score(y_val, val_pred, zero_division=0),
    }

    print("Train metrics:")
    for k, v in train_metrics.items():
        print(f"{k}: {v:.4f}")
    print("\nValidation metrics:")
    for k, v in val_metrics.items():
        print(f"{k}: {v:.4f}")
    print("\nConfusion matrix:")
    print(confusion_matrix(y_val, val_pred))
    print("\nClassification report:")
    print(classification_report(y_val, val_pred, zero_division=0))

    return model, train_metrics, val_metrics


def save_model(model: CatBoostClassifier, path: Path | None = None):
    """
    Save trained model.
    """
    path = path or (PROCESSED_DIR / "models" / "catboost.joblib")
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    print(f"Model saved to: {path}")


if __name__ == "__main__":
    X_train, y_train, X_val, y_val = load_splits()
    model, _, _ = train_and_evaluate(X_train, y_train, X_val, y_val, threshold=0.55)
    save_model(model)
