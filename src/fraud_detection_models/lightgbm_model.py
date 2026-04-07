# LightGBM Model — Standalone Training & Evaluation

import joblib
import numpy as np
import pandas as pd
import lightgbm as lgb
from pathlib import Path
from sklearn.metrics import (
    roc_auc_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
)

BASE_DIR   = Path(__file__).resolve().parents[2]
SPLITS_DIR = BASE_DIR / "data" / "processed" / "splits"
MODELS_DIR = BASE_DIR / "data" / "processed" / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# LightGBM hyperparameters
# scale_pos_weight: controls how much the model penalises missing fraud cases
# Too high (27) → model overcorrects and fails to learn
# Start at 7 — can tune upward if recall is too low

LGBM_PARAMS = {
    "objective":          "binary",
    "metric":             ["auc", "binary_logloss"],
    "boosting_type":      "gbdt",
    "n_estimators":       1000,
    "learning_rate":      0.005,     # slower — forces more rounds
    "num_leaves":         127,       # increased from 63 — more complex trees
    "max_depth":          -1,
    "min_child_samples":  10,        # reduced from 20
    "feature_fraction":   0.8,
    "bagging_fraction":   0.8,
    "bagging_freq":       5,
    "reg_alpha":          0.1,
    "reg_lambda":         0.1,
    "scale_pos_weight":   1,         # start with no weighting — test baseline
    "random_state":       42,
    "n_jobs":             -1,
    "verbose":            -1,
}

def load_splits():
    # Load the preprocessed train/val splits saved by the preprocessing pipeline.

    # Why load from disk rather than re-running the pipeline? Preprocessing takes
    # ~15-20 minutes on the full dataset. Saving the splits as joblib files lets
    # model training scripts start immediately without re-running the pipeline,
    # and ensures all four models are trained on exactly the same data.
   
    print("Loading preprocessed splits from disk...")
    X_train = joblib.load(SPLITS_DIR / "X_train.joblib")
    y_train = joblib.load(SPLITS_DIR / "y_train.joblib")
    X_val   = joblib.load(SPLITS_DIR / "X_val.joblib")
    y_val   = joblib.load(SPLITS_DIR / "y_val.joblib")

    # Drop TransactionID — it is a unique row identifier, not a predictive feature
    # Keeping it confuses the model as it learns spurious ID-based patterns
    for df in [X_train, X_val]:
        if "TransactionID" in df.columns:
            df.drop(columns=["TransactionID"], inplace=True)
            print("  Dropped TransactionID")

    # Convert all columns to standard float32 — fixes PyArrow dtype issues
    print("  Converting dtypes to float32...")
    X_train = X_train.astype({col: "float32" for col in X_train.columns})
    X_val   = X_val.astype({col: "float32" for col in X_val.columns})
    y_train = y_train.astype(int)
    y_val   = y_val.astype(int)

    print(f"  X_train : {X_train.shape} | fraud rate: {y_train.mean():.4f}")
    print(f"  X_val   : {X_val.shape}   | fraud rate: {y_val.mean():.4f}")
    print(f"  Dtypes  : {X_train.dtypes.unique()}")
    return X_train, y_train, X_val, y_val

def train(X_train, y_train, X_val, y_val):
    # Train LightGBM with early stopping on validation AUC.
    # Low learning rate (0.005) with 1000 estimators lets the model converge gradually, faster  than XGBoost on tabular data due to leaf-wise tree growth.
    print("\nTraining LightGBM model...")

    model = lgb.LGBMClassifier(**LGBM_PARAMS)

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric="auc",
        callbacks=[
            lgb.early_stopping(stopping_rounds=100, verbose=True),
            lgb.log_evaluation(period=100),
        ]
    )

    print(f"\nBest iteration: {model.best_iteration_}")
    return model

def evaluate(model, X_val, y_val, threshold=0.5):
    # Evaluate on the validation set. Reports ROC-AUC, PR-AUC, recall, precision, and F1.
    # Also sweeps thresholds to find the one that maximises F1 — more meaningful than default
    # 0.5 when the fraud class is only ~3.5% of the data.

    print("LIGHTGBM EVALUATION RESULTS")
   
    # Predicted probabilities and labels
    y_prob = model.predict_proba(X_val)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)

    # Core metrics
    auc    = roc_auc_score(y_val, y_prob)
    pr_auc = average_precision_score(y_val, y_prob)
    recall = recall_score(y_val, y_pred, zero_division=0)
    prec   = precision_score(y_val, y_pred, zero_division=0)
    f1     = f1_score(y_val, y_pred, zero_division=0)

    print(f"\nThreshold : {threshold}")
    print(f"ROC-AUC   : {auc:.4f}   ← primary metric (target: 0.975+)")
    print(f"PR-AUC    : {pr_auc:.4f}  ← important for imbalanced data")
    print(f"Recall    : {recall:.4f}  ← fraud cases caught (target: 0.76+)")
    print(f"Precision : {prec:.4f}  ← of flagged, how many are fraud")
    print(f"F1 Score  : {f1:.4f}")

    # Confusion matrix
    cm = confusion_matrix(y_val, y_pred)
    tn, fp, fn, tp = cm.ravel()
    print(f"\nConfusion Matrix:")
    print(f"  True Negatives  (legit correctly cleared): {tn:,}")
    print(f"  False Positives (legit flagged as fraud):  {fp:,}")
    print(f"  False Negatives (fraud missed):            {fn:,}")
    print(f"  True Positives  (fraud correctly caught):  {tp:,}")

    # Full classification report
    print(f"\nClassification Report:")
    print(classification_report(
        y_val, y_pred,
        target_names=["Legit", "Fraud"],
        zero_division=0
    ))

    # Top 20 most important features
    print("\nTop 20 Feature Importances:")
    importance = pd.Series(
        model.feature_importances_,
        index=X_val.columns
    ).sort_values(ascending=False)
    print(importance.head(20).to_string())

    from sklearn.metrics import precision_recall_curve

    # Find optimal threshold — maximises F1 score
    precisions, recalls, thresholds = precision_recall_curve(y_val, y_prob)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
    best_threshold = thresholds[np.argmax(f1_scores)]
    print(f"\nOptimal threshold (max F1): {best_threshold:.4f}")

    # Re-evaluate with optimal threshold
    y_pred_optimal = (y_prob >= best_threshold).astype(int)
    print(f"Recall at optimal threshold  : {recall_score(y_val, y_pred_optimal, zero_division=0):.4f}")
    print(f"Precision at optimal threshold: {precision_score(y_val, y_pred_optimal, zero_division=0):.4f}")

    return {
        "auc":       auc,
        "pr_auc":    pr_auc,
        "recall":    recall,
        "precision": prec,
        "f1":        f1,
        "y_prob":    y_prob,
    }

def save_model(model, path=None):
    path = Path(path or MODELS_DIR / "lightgbm_model.joblib")
    joblib.dump(model, path)
    print(f"\nModel saved to {path}")
    return path

def main():
    X_train, y_train, X_val, y_val = load_splits()
    model   = train(X_train, y_train, X_val, y_val)
    results = evaluate(model, X_val, y_val, threshold=0.2)
    save_model(model)

    print(f"SUMMARY: AUC={results['auc']:.4f} | "
          f"Recall={results['recall']:.4f} | "
          f"Precision={results['precision']:.4f}")
    return model, results

if __name__ == "__main__":
    main()