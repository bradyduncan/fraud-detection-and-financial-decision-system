import sys
from pathlib import Path

import joblib
import pandas as pd
from xgboost import XGBClassifier
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
from src.utils.data_splits import load_splits
from src.fraud_detection_models.experimentation import test_scale_pos_weights

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
    Train XGBoost on train set and evaluate on validation set.
    """
    params = {
        "n_estimators": 4000,
        "max_depth": 5,
        "learning_rate": .1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "objective": "binary:logistic",
        "eval_metric": "aucpr",
        "random_state": RANDOM_SEED,
        "tree_method": "hist",
        "early_stopping_rounds": 20,
        "min_child_weight": 5,
        "gamma": .5,
    }
    if "scale_pos_weight" not in params:
        pos = int((y_train == 1).sum())
        neg = int((y_train == 0).sum())
        if pos > 0:
            # highest precision: *.35, highest recall: *.65, highest f1: *.4
            params["scale_pos_weight"] = .65 * neg / pos
    if model_params:
        params.update(model_params)

    model = XGBClassifier(**params)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

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


def save_model(model: XGBClassifier, path: Path | None = None):
    """
    Save trained model to disk.
    """
    path = path or (PROCESSED_DIR / "models" / "xgboost.joblib")
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    print(f"Model saved to: {path}")


if __name__ == "__main__":
    X_train, y_train, X_val, y_val = load_splits()
    model, _, _ = train_and_evaluate(X_train, y_train, X_val, y_val, threshold=0.4)
    save_model(model)
