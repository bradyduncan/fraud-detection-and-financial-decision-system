import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import (
    roc_auc_score,
    recall_score,
    precision_score,
    f1_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
    precision_recall_curve,
)

BASE_DIR   = Path(__file__).resolve().parents[2]
SPLITS_DIR = BASE_DIR / "data" / "processed" / "splits"
MODELS_DIR = BASE_DIR / "data" / "processed" / "models"
OUTPUT_DIR = BASE_DIR / "data" / "processed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Ensemble weights (must sum to 1.0)
ENSEMBLE_WEIGHTS = {
    "xgboost":   0.25,   # highest recall (0.73 on val set)
    "lightgbm":  0.40,   # best precision (0.53 on val set)
    "catboost":  0.35,   # middle ground, adds diversity
}


# Demo card users 
DEMO_USERS = {
    "Alice": {
        "Card 1": {"card1": 7919,  "card4": "mastercard", "card6": "debit"},
        "Card 2": {"card1": 15066, "card4": "mastercard", "card6": "credit"},
    },
    "Bob": {
        "Card 1": {"card1": 9500,  "card4": "visa", "card6": "debit"},
        "Card 2": {"card1": 6019,  "card4": "visa", "card6": "credit"},
    },
    "Carol": {
        "Card 1": {"card1": 15885, "card4": "visa", "card6": "debit"},
        "Card 2": {"card1": 7585,  "card4": "visa", "card6": "credit"},
    },
    "Dave": {
        "Card 1": {"card1": 9633,  "card4": "visa", "card6": "credit"},
        "Card 2": {"card1": 12695, "card4": "visa", "card6": "debit"},
    },
}

def load_data():
    # Load validation splits from preprocessing pipeline output.
    print("Loading validation splits...")
    X_val = joblib.load(SPLITS_DIR / "X_val.joblib")
    y_val = joblib.load(SPLITS_DIR / "y_val.joblib")
    print(f"  X_val shape : {X_val.shape}")
    print(f"  Fraud rate  : {y_val.mean():.4f}")
    return X_val, y_val

def prepare_features(X_val):
    # Prepare features for model inference.
    
    card_cols = ["card1", "card2", "card3", "card4", "card5", "card6"]
    meta_cols = ["TransactionDT", "TransactionAmt", "ProductCD",
                 "P_emaildomain", "addr1"]

    # Save card + transaction metadata before dropping anything
    meta = X_val[[c for c in card_cols + meta_cols if c in X_val.columns]].copy()

    X_model = X_val.copy()
    if "TransactionID" in X_model.columns:
        X_model = X_model.drop(columns=["TransactionID"])

    X_model = X_model.astype({col: "float32" for col in X_model.columns})
    return X_model, meta


def load_models():
    # Load all three trained models from disk.
    print("\nLoading ensemble models...")
    models = {}

    model_files = {
        "lightgbm": MODELS_DIR / "lightgbm_model.joblib",
        "xgboost":  MODELS_DIR / "xgboost.joblib",
        "catboost": MODELS_DIR / "catboost.joblib",
    }

    for name, path in model_files.items():
        if not path.exists():
            raise FileNotFoundError(
                f"Model file not found: {path}\n"
                f"Make sure you have run {name}_model.py first."
            )
        models[name] = joblib.load(path)
        print(f"  Loaded {name} from {path.name}")

    return models


def generate_ensemble_predictions(models, X_model, y_val):
    # Generate weighted ensemble probability scores.

    print("\nGenerating ensemble predictions...")

    individual_probs = {}
    for name, model in models.items():
        proba = model.predict_proba(X_model)[:, 1]
        individual_probs[name] = proba
        print(f"  {name:10s} → mean fraud prob: {proba.mean():.4f}  "
              f"(weight: {ENSEMBLE_WEIGHTS[name]:.0%})")

    # Weighted average of probabilities
    ensemble_proba = sum(
        individual_probs[name] * ENSEMBLE_WEIGHTS[name]
        for name in ENSEMBLE_WEIGHTS
    )

    print(f"\n Ensemble → mean fraud prob: {np.array(ensemble_proba).mean():.4f}")
    return ensemble_proba, individual_probs


# def find_optimal_threshold(ensemble_proba, y_val):
#     # Find the threshold that maximises recall while keeping precision >= 0.25.
   
#     print("\nFinding optimal threshold...")

#     precisions, recalls, thresholds = precision_recall_curve(y_val, ensemble_proba)

#     # Find threshold that maximises recall with precision >= 0.25
#     best_threshold = THRESHOLD
#     best_recall    = 0.0

#     for prec, rec, thresh in zip(precisions[:-1], recalls[:-1], thresholds):
#         if prec >= 0.25 and rec > best_recall:
#             best_recall    = rec
#             best_threshold = thresh

#     # Also compute max-F1 threshold for reference
#     f1_scores     = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
#     best_f1_thresh = thresholds[np.argmax(f1_scores[:-1])]

#     print(f"  Optimal threshold (max recall, precision>=0.25): {best_threshold:.4f}")
#     print(f"  Reference threshold (max F1)                   : {best_f1_thresh:.4f}")
#     print(f"  Using threshold: {best_threshold:.4f}")

#     return 

def find_optimal_threshold(ensemble_proba, y_val):
    threshold = 0.2920
    print(f"\nUsing fixed threshold: {threshold}")
    return threshold

def evaluate_ensemble(ensemble_proba, y_val, threshold):
    # Print full evaluation metrics for the ensemble.
    print("ENSEMBLE EVALUATION RESULTS")   

    y_pred = (ensemble_proba >= threshold).astype(int)

    auc    = roc_auc_score(y_val, ensemble_proba)
    pr_auc = average_precision_score(y_val, ensemble_proba)
    recall = recall_score(y_val, y_pred, zero_division=0)
    prec   = precision_score(y_val, y_pred, zero_division=0)
    f1     = f1_score(y_val, y_pred, zero_division=0)

    print(f"\nThreshold : {threshold:.4f}")
    print(f"ROC-AUC   : {auc:.4f}   ← primary metric (target: 0.975+)")
    print(f"PR-AUC    : {pr_auc:.4f}  ← important for imbalanced data")
    print(f"Recall    : {recall:.4f}  ← fraud cases caught (target: 0.76+)")
    print(f"Precision : {prec:.4f}  ← of flagged, how many are fraud")
    print(f"F1 Score  : {f1:.4f}")

    cm = confusion_matrix(y_val, y_pred)
    tn, fp, fn, tp = cm.ravel()
    print(f"\nConfusion Matrix:")
    print(f"  True Negatives  (legit correctly cleared): {tn:,}")
    print(f"  False Positives (legit flagged as fraud):  {fp:,}")
    print(f"  False Negatives (fraud missed):            {fn:,}")
    print(f"  True Positives  (fraud correctly caught):  {tp:,}")

    print(f"\nClassification Report:")
    print(classification_report(
        y_val, y_pred,
        target_names=["Legit", "Fraud"],
        zero_division=0
    ))

    return {"auc": auc, "pr_auc": pr_auc, "recall": recall,
            "precision": prec, "f1": f1}


def build_predictions_csv(X_val, y_val, ensemble_proba, individual_probs,
                           threshold, meta):
    # Assemble final predictions DataFrame and save to CSV for dashboard.
    print("\nBuilding predictions CSV...")

    df = meta.copy()
    df["isFraud"]              = y_val.values
    df["fraud_probability"]    = ensemble_proba
    df["model_decision"]       = (ensemble_proba >= threshold).astype(int)

    # Individual model probabilities — useful for SHAP and dashboard explainability
    df["prob_lightgbm"]        = individual_probs["lightgbm"]
    df["prob_xgboost"]         = individual_probs["xgboost"]
    df["prob_catboost"]        = individual_probs["catboost"]

    # Calendar date for dashboard display
    BASE_DATE = pd.Timestamp("2020-01-01")
    if "TransactionDT" in df.columns:
        df["transaction_date"] = (
            BASE_DATE + pd.to_timedelta(df["TransactionDT"], unit="s")
        )
    else:
        # TransactionDT was dropped in preprocessing — generate synthetic dates
        # Spread transactions evenly across a 6-month window for dashboard display
        df["transaction_date"] = pd.date_range(
            start="2020-01-01", periods=len(df), freq="30min"
        )

    # Demo user labels for dashboard filtering
    df["demo_user"] = "Unknown"
    df["demo_card"] = "Unknown"

    for user, cards in DEMO_USERS.items():
        for card_label, card_info in cards.items():
            mask = df["card1"] == card_info["card1"]
            df.loc[mask, "demo_user"] = user
            df.loc[mask, "demo_card"] = card_label

    # Save
    output_path = OUTPUT_DIR / "predictions.csv"
    df.to_csv(output_path, index=False)

    print(f"\n  Saved to : {output_path}")
    print(f"  Shape    : {df.shape}")
    print(f"  Columns  : {df.columns.tolist()}")

    # Demo user summary
    print("\nDemo user transaction counts:")
    summary = df[df["demo_user"] != "Unknown"].groupby(
        ["demo_user", "demo_card"]
    ).agg(
        transactions=("fraud_probability", "count"),
        fraud_flagged=("model_decision", "sum"),
        actual_fraud=("isFraud", "sum")
    )
    print(summary.to_string())

    return df


def main():
    print("FRAUD DETECTION — ENSEMBLE PREDICTION PIPELINE")
    print(f"Weights: XGBoost={ENSEMBLE_WEIGHTS['xgboost']:.0%}  "
          f"LightGBM={ENSEMBLE_WEIGHTS['lightgbm']:.0%}  "
          f"CatBoost={ENSEMBLE_WEIGHTS['catboost']:.0%}")

    # 1. Load data
    X_val, y_val = load_data()

    # 2. Prepare features
    X_model, meta = prepare_features(X_val)

    # 3. Load models
    models = load_models()

    # 4. Generate ensemble probabilities
    ensemble_proba, individual_probs = generate_ensemble_predictions(
        models, X_model, y_val
    )

    # 5. Find optimal threshold
    threshold = find_optimal_threshold(ensemble_proba, y_val)

    # 6. Evaluate ensemble
    metrics = evaluate_ensemble(ensemble_proba, y_val, threshold)

    # 7. Save predictions CSV for dashboard
    predictions_df = build_predictions_csv(
        X_val, y_val, ensemble_proba, individual_probs, threshold, meta
    )

    print(f"ENSEMBLE SUMMARY:")
    print(f"  AUC       : {metrics['auc']:.4f}")
    print(f"  Recall    : {metrics['recall']:.4f}")
    print(f"  Precision : {metrics['precision']:.4f}")
    print(f"  F1        : {metrics['f1']:.4f}")
    print(f"  Threshold : {threshold:.4f}")
    print("\npredictions.csv ready for dashboard!")

    return predictions_df


if __name__ == "__main__":
    main()