import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import f1_score, precision_score, recall_score

from src.config import RANDOM_SEED


def sweep_thresholds(
    y_true: pd.Series,
    y_proba: np.ndarray,
    thresholds: list[float] | None = None,
):
    """
    Sweep thresholds and report precision/recall/F1.
    """
    if thresholds is None:
        thresholds = [round(t, 2) for t in np.arange(0.05, 0.96, 0.05)]

    rows = []
    for t in thresholds:
        y_pred = (y_proba >= t).astype(int)
        rows.append(
            {
                "threshold": t,
                "precision": precision_score(y_true, y_pred, zero_division=0),
                "recall": recall_score(y_true, y_pred, zero_division=0),
                "f1": f1_score(y_true, y_pred, zero_division=0),
            }
        )

    sweep_df = pd.DataFrame(rows).sort_values("threshold")
    best_row = sweep_df.sort_values("f1", ascending=False).iloc[0]

    print("\nThreshold sweep (precision/recall/F1):")
    print(sweep_df.to_string(index=False))
    print(
        f"\nBest F1 at threshold={best_row['threshold']:.2f} "
        f"(precision={best_row['precision']:.4f}, "
        f"recall={best_row['recall']:.4f}, f1={best_row['f1']:.4f})"
    )

    return sweep_df


def test_scale_pos_weights(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    weights: list[float],
    threshold: float = 0.7,
    base_params: dict | None = None,
):
    """
    Train XGBoost models across scale_pos_weight values and report metrics.
    """
    params = {
        "n_estimators": 300,
        "max_depth": 6,
        "learning_rate": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "random_state": RANDOM_SEED,
        "tree_method": "hist",
    }
    if base_params:
        params.update(base_params)

    rows = []
    for w in weights:
        params_w = dict(params)
        params_w["scale_pos_weight"] = w
        model = XGBClassifier(**params_w)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

        val_proba = model.predict_proba(X_val)[:, 1]
        val_pred = (val_proba >= threshold).astype(int)

        rows.append(
            {
                "scale_pos_weight": w,
                "precision": precision_score(y_val, val_pred, zero_division=0),
                "recall": recall_score(y_val, val_pred, zero_division=0),
                "f1": f1_score(y_val, val_pred, zero_division=0),
            }
        )

    results_df = pd.DataFrame(rows).sort_values("scale_pos_weight")
    print("\nscale_pos_weight sweep (precision/recall/F1):")
    print(results_df.to_string(index=False))
    best_row = results_df.sort_values("f1", ascending=False).iloc[0]
    print(
        f"\nBest F1 at scale_pos_weight={best_row['scale_pos_weight']:.4f} "
        f"(precision={best_row['precision']:.4f}, "
        f"recall={best_row['recall']:.4f}, f1={best_row['f1']:.4f})"
    )

    results_list = [
        (row["scale_pos_weight"], row["precision"], row["recall"], row["f1"])
        for row in rows
    ]

    return results_list
