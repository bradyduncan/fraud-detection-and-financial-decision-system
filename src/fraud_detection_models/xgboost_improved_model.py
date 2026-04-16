import sys
from pathlib import Path

import joblib
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
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

__all__ = [
    "train_stage1_model",
    "build_stage2_dataset",
    "train_stage2_model",
    "tune_stage1_threshold",
    "tune_stage2_threshold",
    "tune_cascade_thresholds",
    "predict_cascade",
    "evaluate_cascade",
    "FraudCascadeModel",
    "save_cascade_model",
    "load_cascade_model",
    "load_splits",
]


def _safe_roc_auc(y_true: pd.Series, y_proba: np.ndarray) -> float:
    try:
        return roc_auc_score(y_true, y_proba)
    except ValueError:
        return float("nan")


def _safe_pr_auc(y_true: pd.Series, y_proba: np.ndarray) -> float:
    try:
        return average_precision_score(y_true, y_proba)
    except ValueError:
        return float("nan")


def _compute_metrics(y_true: pd.Series, y_pred: np.ndarray, y_proba: np.ndarray) -> dict:
    return {
        "roc_auc": _safe_roc_auc(y_true, y_proba),
        "pr_auc": _safe_pr_auc(y_true, y_proba),
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, y_pred),
    }


def _threshold_candidates_from_proba(y_proba: np.ndarray) -> np.ndarray:
    # Generate candidate thresholds from unique predicted probabilities, clipped to [0, 1].
    scores = np.asarray(y_proba, dtype=float)
    scores = scores[np.isfinite(scores)]
    if scores.size == 0:
        return np.array([0.5], dtype=float)
    candidates = np.unique(scores)
    return np.clip(candidates, 0.0, 1.0)


def _add_stage1_meta_features(
    X: pd.DataFrame,
    stage1_proba: np.ndarray,
    stage1_threshold: float | None = None,
    add_bucket: bool = True,
    add_rank: bool = True,
) -> pd.DataFrame:
    X_meta = X.copy()
    X_meta["stage1_proba"] = stage1_proba
    eps = 1e-6
    clipped = np.clip(stage1_proba, eps, 1.0 - eps)
    X_meta["stage1_logit"] = np.log(clipped / (1.0 - clipped))
    if stage1_threshold is not None:
        X_meta["stage1_margin"] = stage1_proba - stage1_threshold

    if add_bucket:
        bins = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        X_meta["stage1_bucket"] = np.digitize(stage1_proba, bins, right=True)

    if add_rank:
        ranks = pd.Series(stage1_proba, index=X.index).rank(method="average", pct=True)
        X_meta["stage1_rank_pct"] = ranks.to_numpy()

    return X_meta


def train_stage1_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    model_params: dict | None = None,
) -> XGBClassifier:
    # Train Stage 1 XGBoost model to identify potential frauds with high recall.
    params = {
        "n_estimators": 4000,
        "max_depth": 4,
        "learning_rate": .1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "objective": "binary:logistic",
        "eval_metric": "aucpr",
        "random_state": RANDOM_SEED,
        "tree_method": "hist",
        "early_stopping_rounds": 20,
        "min_child_weight": 10,
        "gamma": 1,
        "reg_alpha": 1,
        "reg_lambda": 1,
    }
    if "scale_pos_weight" not in params:
        pos = int((y_train == 1).sum())
        neg = int((y_train == 0).sum())
        if pos > 0:
            params["scale_pos_weight"] = 1.0 * neg / pos
    if model_params:
        params.update(model_params)

    model = XGBClassifier(**params)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    return model


def tune_stage1_threshold(
    y_true: pd.Series,
    y_proba: np.ndarray,
    recall_target: float | None = None,
    min_recall: float = 0.9,
    thresholds: np.ndarray | None = None,
) -> tuple[float, pd.DataFrame]:
    #   Tune Stage 1 threshold to achieve target recall while maximizing precision/F1.
    thresholds = thresholds if thresholds is not None else _threshold_candidates_from_proba(y_proba)
    records = []
    for thresh in thresholds:
        preds = (y_proba >= thresh).astype(int)
        metrics = _compute_metrics(y_true, preds, y_proba)
        records.append(
            {
                "threshold": float(thresh),
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1"],
            }
        )
    results = pd.DataFrame(records)

    if recall_target is not None:
        eligible = results[results["recall"] >= recall_target]
        if not eligible.empty:
            best = eligible.sort_values(["precision", "f1"], ascending=False).iloc[0]
            return float(best["threshold"]), results

    eligible = results[results["recall"] >= min_recall]
    if not eligible.empty:
        best = eligible.sort_values(["precision", "f1"], ascending=False).iloc[0]
        return float(best["threshold"]), results

    best = results.sort_values(["recall", "precision", "f1"], ascending=False).iloc[0]
    return float(best["threshold"]), results


def build_stage2_dataset(
    X: pd.DataFrame,
    y: pd.Series,
    stage1_proba: np.ndarray,
    stage1_pred: np.ndarray,
    stage1_threshold: float | None = None,
    add_meta_features: bool = True,
) -> tuple[pd.DataFrame, pd.Series]:
    # Build Stage 2 training dataset from Stage 1 positives, optionally adding Stage 1 meta features.
    mask = stage1_pred.astype(bool)
    if mask.sum() == 0:
        raise ValueError("Stage 1 produced zero positives; cannot train Stage 2.")
    X_stage2 = X.loc[mask].copy()
    y_stage2 = y.loc[mask].copy()

    if add_meta_features:
        X_stage2 = _add_stage1_meta_features(
            X_stage2,
            stage1_proba[mask],
            stage1_threshold=stage1_threshold,
        )

    return X_stage2, y_stage2


def _build_stage2_model(
    model_type: str = "random_forest",
    model_params: dict | None = None,
):
    if model_type == "logreg":
        params = {
            "solver": "liblinear",
            "max_iter": 1000,
            "class_weight": "balanced",
        }
        if model_params:
            params.update(model_params)
        return LogisticRegression(**params)
    if model_type == "random_forest":
        params = {
            "n_estimators": 300,
            "max_depth": None,
            "random_state": RANDOM_SEED,
            "n_jobs": -1,
            "class_weight": "balanced",
        }
        if model_params:
            params.update(model_params)
        return RandomForestClassifier(**params)
    if model_type == "xgboost":
        params = {
            "n_estimators": 800,
            "max_depth": 3,
            "learning_rate": 0.1,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "objective": "binary:logistic",
            "eval_metric": "aucpr",
            "random_state": RANDOM_SEED,
            "tree_method": "hist",
        }
        if model_params:
            params.update(model_params)
        return XGBClassifier(**params)

    raise ValueError(f"Unsupported stage2 model_type: {model_type}")


def train_stage2_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    model_type: str = "random_forest",
    model_params: dict | None = None,
    calibrate: bool = False,
    calibration_cv: int = 3,
    sample_weight: np.ndarray | None = None,
):
    # Train Stage 2 model to refine Stage 1 positives, with optional calibration and sample weighting.
    if model_type == "xgboost":
        params = dict(model_params) if model_params else {}
        if "scale_pos_weight" not in params:
            pos = int((y_train == 1).sum())
            neg = int((y_train == 0).sum())
            if pos > 0:
                params["scale_pos_weight"] = neg / pos
        base_model = _build_stage2_model(model_type=model_type, model_params=params)
    else:
        base_model = _build_stage2_model(model_type=model_type, model_params=model_params)
    if calibrate:
        try:
            model = CalibratedClassifierCV(estimator=base_model, method="sigmoid", cv=calibration_cv)
        except TypeError:
            model = CalibratedClassifierCV(base_estimator=base_model, method="sigmoid", cv=calibration_cv)
    else:
        model = base_model

    model.fit(X_train, y_train, sample_weight=sample_weight)
    return model


def generate_oof_stage1_proba(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    stage1_params: dict | None = None,
    n_splits: int = 5,
) -> np.ndarray:
    # Generate out-of-fold predicted probabilities from Stage 1 model for the entire training set, using Stratified K-Fold.
    oof_proba = np.zeros(len(X_train), dtype=float)
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_SEED)

    for train_idx, val_idx in splitter.split(X_train, y_train):
        X_tr = X_train.iloc[train_idx]
        y_tr = y_train.iloc[train_idx]
        X_va = X_train.iloc[val_idx]
        y_va = y_train.iloc[val_idx]

        model = train_stage1_model(X_tr, y_tr, X_va, y_va, model_params=stage1_params)
        oof_proba[val_idx] = model.predict_proba(X_va)[:, 1]

    return oof_proba


def tune_stage2_threshold(
    y_true: pd.Series,
    stage1_pred: np.ndarray,
    stage2_proba: np.ndarray,
    stage1_threshold: float,
    min_cascade_recall: float | None = None,
    thresholds: np.ndarray | None = None,
) -> tuple[float, pd.DataFrame]:
    #  Tune Stage 2 threshold to maximize F1 while optionally enforcing a minimum recall on the cascade. Only evaluate Stage 2 thresholds on the subset of data routed to Stage 2 by Stage 1.
    if len(stage2_proba) == len(stage1_pred):
        stage2_proba_subset = stage2_proba[stage1_pred == 1]
    else:
        stage2_proba_subset = stage2_proba

    thresholds = thresholds if thresholds is not None else _threshold_candidates_from_proba(stage2_proba_subset)
    records = []
    for thresh in thresholds:
        stage2_pred = (stage2_proba_subset >= thresh).astype(int)
        final_pred = np.zeros_like(stage1_pred)
        final_pred[stage1_pred == 1] = stage2_pred
        metrics = _compute_metrics(y_true, final_pred, stage2_proba)
        records.append(
            {
                "stage1_threshold": float(stage1_threshold),
                "threshold": float(thresh),
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1"],
            }
        )
    results = pd.DataFrame(records)

    if min_cascade_recall is not None:
        eligible = results[results["recall"] >= min_cascade_recall]
        if not eligible.empty:
            best = eligible.sort_values(["f1", "precision"], ascending=False).iloc[0]
            return float(best["threshold"]), results

    best = results.sort_values(["f1", "precision"], ascending=False).iloc[0]
    return float(best["threshold"]), results


def tune_cascade_thresholds(
    X_val: pd.DataFrame,
    y_val: pd.Series,
    stage1_model: XGBClassifier,
    stage2_model,
    stage1_thresholds: np.ndarray | None = None,
    stage2_thresholds: np.ndarray | None = None,
    min_cascade_recall: float | None = None,
    add_meta_features: bool = True,
) -> tuple[float, float, pd.DataFrame]:
    # jointly tune Stage 1 and Stage 2 thresholds to maximize cascade F1 while optionally enforcing a minimum recall on the cascade. Evaluates all combinations of provided Stage 1 and Stage 2 thresholds.
    stage1_thresholds = stage1_thresholds if stage1_thresholds is not None else np.linspace(0.01, 0.99, 50)
    stage2_thresholds = stage2_thresholds if stage2_thresholds is not None else np.linspace(0.01, 0.5, 50)
    records = []

    stage1_proba = stage1_model.predict_proba(X_val)[:, 1]
    for t1 in stage1_thresholds:
        stage1_pred = (stage1_proba >= t1).astype(int)
        routed_mask = stage1_pred.astype(bool)

        stage2_proba_full = np.full(len(X_val), np.nan)
        if routed_mask.any():
            X_stage2 = X_val.loc[routed_mask].copy()
            if add_meta_features:
                X_stage2 = _add_stage1_meta_features(
                    X_stage2,
                    stage1_proba[routed_mask],
                    stage1_threshold=float(t1),
                )
            stage2_proba_full[routed_mask] = stage2_model.predict_proba(X_stage2)[:, 1]

        stage2_proba_subset = stage2_proba_full[routed_mask] if routed_mask.any() else np.array([])

        for t2 in stage2_thresholds:
            if routed_mask.any():
                stage2_pred = (stage2_proba_subset >= t2).astype(int)
            else:
                stage2_pred = np.array([], dtype=int)
            final_pred = np.zeros_like(stage1_pred)
            if routed_mask.any():
                final_pred[routed_mask] = stage2_pred
            metrics = _compute_metrics(y_val, final_pred, np.nan_to_num(stage2_proba_full, nan=0.0))
            records.append(
                {
                    "stage1_threshold": float(t1),
                    "stage2_threshold": float(t2),
                    "precision": metrics["precision"],
                    "recall": metrics["recall"],
                    "f1": metrics["f1"],
                }
            )

    results = pd.DataFrame(records)
    if min_cascade_recall is not None:
        eligible = results[results["recall"] >= min_cascade_recall]
        if not eligible.empty:
            best = eligible.sort_values(["f1", "precision"], ascending=False).iloc[0]
            return float(best["stage1_threshold"]), float(best["stage2_threshold"]), results

    best = results.sort_values(["f1", "precision"], ascending=False).iloc[0]
    return float(best["stage1_threshold"]), float(best["stage2_threshold"]), results


def predict_cascade(
    X: pd.DataFrame,
    stage1_model: XGBClassifier,
    stage2_model,
    stage1_threshold: float,
    stage2_threshold: float,
    add_meta_features: bool = True,
) -> dict:
    # predict with the cascade: apply Stage 1 to all data, route positives to Stage 2, and combine predictions. Returns detailed outputs including probabilities, predictions, and routing mask.
    stage1_proba = stage1_model.predict_proba(X)[:, 1]
    stage1_pred = (stage1_proba >= stage1_threshold).astype(int)

    routed_mask = stage1_pred.astype(bool)
    stage2_proba_full = np.full(len(X), np.nan)
    stage2_pred_full = np.zeros(len(X), dtype=int)

    if routed_mask.any():
        X_stage2 = X.loc[routed_mask].copy()
        if add_meta_features:
            X_stage2 = _add_stage1_meta_features(
                X_stage2,
                stage1_proba[routed_mask],
                stage1_threshold=stage1_threshold,
            )
        stage2_proba = stage2_model.predict_proba(X_stage2)[:, 1]
        stage2_pred = (stage2_proba >= stage2_threshold).astype(int)
        stage2_proba_full[routed_mask] = stage2_proba
        stage2_pred_full[routed_mask] = stage2_pred

    final_pred = stage2_pred_full.copy()
    cascade_proba = np.where(routed_mask, stage2_proba_full, 0.0)

    return {
        "stage1_proba": stage1_proba,
        "stage1_pred": stage1_pred,
        "stage2_proba": stage2_proba_full,
        "final_pred": final_pred,
        "cascade_proba": cascade_proba,
        "routed_mask": routed_mask,
    }


def evaluate_cascade(
    X_val: pd.DataFrame,
    y_val: pd.Series,
    stage1_model: XGBClassifier,
    stage2_model,
    stage1_threshold: float,
    stage2_threshold: float,
    add_meta_features: bool = True,
) -> dict:
    #evaluate the cascade on validation data, computing metrics for Stage 1 alone, Stage 2 alone (on routed subset), and the final cascade. Prints detailed metrics and returns a summary dictionary.
    outputs = predict_cascade(
        X_val,
        stage1_model,
        stage2_model,
        stage1_threshold,
        stage2_threshold,
        add_meta_features=add_meta_features,
    )

    stage1_metrics = _compute_metrics(y_val, outputs["stage1_pred"], outputs["stage1_proba"])
    cascade_metrics = _compute_metrics(y_val, outputs["final_pred"], outputs["cascade_proba"])

    routed_mask = outputs["routed_mask"]
    stage2_metrics = {}
    if routed_mask.any():
        y_stage2 = y_val.loc[routed_mask]
        stage2_pred = outputs["final_pred"][routed_mask]
        stage2_proba = outputs["stage2_proba"][routed_mask]
        stage2_metrics = _compute_metrics(y_stage2, stage2_pred, stage2_proba)

    summary = {
        "total_records": int(len(y_val)),
        "stage1_flagged": int(outputs["routed_mask"].sum()),
        "stage1_flagged_pct": float(outputs["routed_mask"].mean() * 100.0),
        "final_fraud_predictions": int(outputs["final_pred"].sum()),
    }

    print("Stage 1 metrics:")
    for k, v in stage1_metrics.items():
        if k == "confusion_matrix":
            continue
        print(f"{k}: {v:.4f}")
    print("\nStage 1 confusion matrix:")
    print(stage1_metrics["confusion_matrix"])

    if stage2_metrics:
        print("\nStage 2 (routed subset) metrics:")
        for k, v in stage2_metrics.items():
            if k == "confusion_matrix":
                continue
            print(f"{k}: {v:.4f}")
        print("\nStage 2 confusion matrix:")
        print(stage2_metrics["confusion_matrix"])

    print("\nFinal cascade metrics:")
    for k, v in cascade_metrics.items():
        if k == "confusion_matrix":
            continue
        print(f"{k}: {v:.4f}")
    print("\nFinal cascade confusion matrix:")
    print(cascade_metrics["confusion_matrix"])

    print("\nSummary:")
    print(f"Total records: {summary['total_records']}")
    print(f"Stage 1 flagged count: {summary['stage1_flagged']} ({summary['stage1_flagged_pct']:.2f}%)")
    print(f"Stage 2 reviewed count: {summary['stage1_flagged']}")
    print(f"Final fraud predictions count: {summary['final_fraud_predictions']}")
    print(f"Stage 1 recall/precision/F1: {stage1_metrics['recall']:.4f} / {stage1_metrics['precision']:.4f} / {stage1_metrics['f1']:.4f}")
    print(f"Final cascade recall/precision/F1: {cascade_metrics['recall']:.4f} / {cascade_metrics['precision']:.4f} / {cascade_metrics['f1']:.4f}")

    return {
        "stage1_metrics": stage1_metrics,
        "stage2_metrics": stage2_metrics,
        "cascade_metrics": cascade_metrics,
        "summary": summary,
        "outputs": outputs,
    }


def save_cascade_model(artifacts: dict, path: Path | None = None):
    # Save cascade artifacts (models, thresholds, tuning results) to disk using joblib.
    path = path or (PROCESSED_DIR / "models" / "xgboost_cascade.joblib")
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifacts, path)
    print(f"Cascade model saved to: {path}")


def load_cascade_model(path: Path | None = None) -> dict:
    # Load cascade artifacts from disk using joblib. Returns a dictionary containing models, thresholds, and tuning results.
    path = path or (PROCESSED_DIR / "models" / "xgboost_cascade.joblib")
    return joblib.load(path)


class FraudCascadeModel:

    def __init__(
        self,
        stage1_params: dict | None = None,
        stage2_model_type: str = "random_forest",
        stage2_params: dict | None = None,
        stage1_threshold: float | None = None,
        stage2_threshold: float | None = None,
        stage1_recall_target: float | None = None,
        stage1_min_recall: float = 0.9,
        stage2_min_cascade_recall: float | None = None,
        use_oof_stage1: bool = True,
        oof_splits: int = 5,
        threshold_tune_split: float = 0.1,
        threshold_tune_random_state: int = RANDOM_SEED,
        hard_negative_weight: float = 2.0,
        calibrate_stage2: bool = False,
        calibration_cv: int = 3,
        add_meta_features: bool = True,
    ):
        self.stage1_params = stage1_params
        self.stage2_model_type = stage2_model_type
        self.stage2_params = stage2_params
        self.stage1_threshold = stage1_threshold
        self.stage2_threshold = stage2_threshold
        self.stage1_recall_target = stage1_recall_target
        self.stage1_min_recall = stage1_min_recall
        self.stage2_min_cascade_recall = stage2_min_cascade_recall
        self.use_oof_stage1 = use_oof_stage1
        self.oof_splits = oof_splits
        self.threshold_tune_split = threshold_tune_split
        self.threshold_tune_random_state = threshold_tune_random_state
        self.hard_negative_weight = hard_negative_weight
        self.calibrate_stage2 = calibrate_stage2
        self.calibration_cv = calibration_cv
        self.add_meta_features = add_meta_features

        self.stage1_model = None
        self.stage2_model = None
        self.stage1_tuning_results = None
        self.stage2_tuning_results = None

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame, y_val: pd.Series):
        if self.threshold_tune_split > 0:
            X_train_main, X_tune, y_train_main, y_tune = train_test_split(
                X_train,
                y_train,
                test_size=self.threshold_tune_split,
                random_state=self.threshold_tune_random_state,
                stratify=y_train,
            )
        else:
            X_train_main, y_train_main = X_train, y_train
            X_tune, y_tune = X_val, y_val

        self.stage1_model = train_stage1_model(
            X_train_main,
            y_train_main,
            X_val,
            y_val,
            model_params=self.stage1_params,
        )

        stage1_val_proba = self.stage1_model.predict_proba(X_val)[:, 1]
        if self.stage1_threshold is None:
            self.stage1_threshold, self.stage1_tuning_results = tune_stage1_threshold(
                y_val,
                stage1_val_proba,
                recall_target=self.stage1_recall_target,
                min_recall=self.stage1_min_recall,
            )

        if self.use_oof_stage1:
            stage1_train_proba = generate_oof_stage1_proba(
                X_train_main,
                y_train_main,
                stage1_params=self.stage1_params,
                n_splits=self.oof_splits,
            )
        else:
            stage1_train_proba = self.stage1_model.predict_proba(X_train_main)[:, 1]
        stage1_train_pred = (stage1_train_proba >= self.stage1_threshold).astype(int)
        X_stage2_train, y_stage2_train = build_stage2_dataset(
            X_train_main,
            y_train_main,
            stage1_train_proba,
            stage1_train_pred,
            stage1_threshold=self.stage1_threshold,
            add_meta_features=self.add_meta_features,
        )

        sample_weight = np.ones(len(y_stage2_train), dtype=float)
        hard_neg_mask = (y_stage2_train == 0).to_numpy()
        sample_weight[hard_neg_mask] = self.hard_negative_weight

        self.stage2_model = train_stage2_model(
            X_stage2_train,
            y_stage2_train,
            model_type=self.stage2_model_type,
            model_params=self.stage2_params,
            calibrate=self.calibrate_stage2,
            calibration_cv=self.calibration_cv,
            sample_weight=sample_weight,
        )

        if self.stage2_threshold is None:
            val_outputs = predict_cascade(
                X_tune,
                self.stage1_model,
                self.stage2_model,
                self.stage1_threshold,
                stage2_threshold=0.5,
                add_meta_features=self.add_meta_features,
            )
            self.stage2_threshold, self.stage2_tuning_results = tune_stage2_threshold(
                y_tune,
                val_outputs["stage1_pred"],
                np.nan_to_num(val_outputs["stage2_proba"], nan=0.0),
                self.stage1_threshold,
                min_cascade_recall=self.stage2_min_cascade_recall,
            )

        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        outputs = predict_cascade(
            X,
            self.stage1_model,
            self.stage2_model,
            self.stage1_threshold,
            self.stage2_threshold,
            add_meta_features=self.add_meta_features,
        )
        return outputs["final_pred"]

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        outputs = predict_cascade(
            X,
            self.stage1_model,
            self.stage2_model,
            self.stage1_threshold,
            self.stage2_threshold,
            add_meta_features=self.add_meta_features,
        )
        return outputs["cascade_proba"]

    def evaluate(self, X_val: pd.DataFrame, y_val: pd.Series) -> dict:
        return evaluate_cascade(
            X_val,
            y_val,
            self.stage1_model,
            self.stage2_model,
            self.stage1_threshold,
            self.stage2_threshold,
            add_meta_features=self.add_meta_features,
        )

    def save(self, path: Path | None = None):
        artifacts = {
            "stage1_model": self.stage1_model,
            "stage2_model": self.stage2_model,
            "stage1_threshold": self.stage1_threshold,
            "stage2_threshold": self.stage2_threshold,
            "stage1_params": self.stage1_params,
            "stage2_model_type": self.stage2_model_type,
            "stage2_params": self.stage2_params,
            "stage1_recall_target": self.stage1_recall_target,
            "stage1_min_recall": self.stage1_min_recall,
            "stage2_min_cascade_recall": self.stage2_min_cascade_recall,
            "use_oof_stage1": self.use_oof_stage1,
            "oof_splits": self.oof_splits,
            "threshold_tune_split": self.threshold_tune_split,
            "threshold_tune_random_state": self.threshold_tune_random_state,
            "hard_negative_weight": self.hard_negative_weight,
            "calibrate_stage2": self.calibrate_stage2,
            "calibration_cv": self.calibration_cv,
            "add_meta_features": self.add_meta_features,
        }
        save_cascade_model(artifacts, path=path)

    @classmethod
    def load(cls, path: Path | None = None):
        artifacts = load_cascade_model(path=path)
        model = cls(
            stage1_params=artifacts.get("stage1_params"),
            stage2_model_type=artifacts.get("stage2_model_type", "logreg"),
            stage2_params=artifacts.get("stage2_params"),
            stage1_threshold=artifacts.get("stage1_threshold"),
            stage2_threshold=artifacts.get("stage2_threshold"),
            stage1_recall_target=artifacts.get("stage1_recall_target"),
            stage1_min_recall=artifacts.get("stage1_min_recall", 0.9),
            stage2_min_cascade_recall=artifacts.get("stage2_min_cascade_recall"),
            use_oof_stage1=artifacts.get("use_oof_stage1", True),
            oof_splits=artifacts.get("oof_splits", 5),
            threshold_tune_split=artifacts.get("threshold_tune_split", 0.1),
            threshold_tune_random_state=artifacts.get("threshold_tune_random_state", RANDOM_SEED),
            hard_negative_weight=artifacts.get("hard_negative_weight", 2.0),
            calibrate_stage2=artifacts.get("calibrate_stage2", False),
            calibration_cv=artifacts.get("calibration_cv", 3),
            add_meta_features=artifacts.get("add_meta_features", True),
        )
        model.stage1_model = artifacts.get("stage1_model")
        model.stage2_model = artifacts.get("stage2_model")
        return model


if __name__ == "__main__":
    X_train, y_train, X_val, y_val = load_splits()
    cascade = FraudCascadeModel(
        stage1_recall_target=0.9,
        stage1_min_recall=0.85,
        stage2_min_cascade_recall=0.75,
        stage2_model_type="xgboost",
        stage2_params={
            "n_estimators": 1200,
            "max_depth": 4,
            "learning_rate": 0.05,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "min_child_weight": 5,
            "gamma": 0,
            "reg_alpha": 0.5,
            "reg_lambda": 1.0,
        },
        use_oof_stage1=True,
        oof_splits=5,
        threshold_tune_split=0.1,
        hard_negative_weight=2.5,
        calibrate_stage2=True,
    )
    cascade.fit(X_train, y_train, X_val, y_val)
    cascade.evaluate(X_val, y_val)
    cascade.save()
