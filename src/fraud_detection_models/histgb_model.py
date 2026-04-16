import sys
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
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
    # Train HistGradientBoostingClassifier with default parameters and class weight adjustment
    params = {
        "loss": "log_loss",
        "learning_rate": 0.05,
        "max_iter": 400,
        "max_depth": 6,
        "min_samples_leaf": 20,
        "l2_regularization": 1.0,
        "max_bins": 255,
        "early_stopping": True,
        "n_iter_no_change": 20,
        "validation_fraction": None,
        "random_state": RANDOM_SEED,
    }
    if "positive_class_weight" not in params:
        pos = int((y_train == 1).sum())
        neg = int((y_train == 0).sum())
        if pos > 0:
            params["positive_class_weight"] = 0.5 * neg / pos
    if model_params:
        params.update(model_params)

    positive_class_weight = float(params.pop("positive_class_weight", 1.0))
    sample_weight = pd.Series(1.0, index=y_train.index, dtype="float64")
    sample_weight.loc[y_train == 1] = positive_class_weight

    model = HistGradientBoostingClassifier(**params)
    model.fit(X_train, y_train, sample_weight=sample_weight)

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


def save_model(model: HistGradientBoostingClassifier, path: Path | None = None):
    path = path or (PROCESSED_DIR / "models" / "histgb.joblib")
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    print(f"Model saved to: {path}")


if __name__ == "__main__":
    X_train, y_train, X_val, y_val = load_splits()
    model, _, _ = train_and_evaluate(X_train, y_train, X_val, y_val, threshold=0.5)
    save_model(model)
