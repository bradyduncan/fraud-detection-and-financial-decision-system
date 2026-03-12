# predict.py — Generate fraud predictions on validation set for dashboard
# Place this file in: src/fraud_detection_models/predict.py

import joblib
import numpy as np
import pandas as pd
from pathlib import Path

# ── Paths (matches your existing lightgbm_model.py structure) ──────────────
BASE_DIR    = Path(__file__).resolve().parents[2]
SPLITS_DIR  = BASE_DIR / "data" / "processed" / "splits"
MODELS_DIR  = BASE_DIR / "data" / "processed" / "models"
OUTPUT_DIR  = BASE_DIR / "data" / "processed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Demo user card1 values (confirmed from EDA) ────────────────────────────
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
}

# Threshold matching your lightgbm evaluation (0.2 used in lightgbm_model.py)
THRESHOLD = 0.2

import joblib
import pandas as pd
from pathlib import Path

# Check exact path being used
splits_dir = Path("data/processed/splits")
print("Path exists:", splits_dir.exists())
print("File exists:", (splits_dir / "X_val.joblib").exists())
print("Pandas version:", pd.__version__)

# Try loading
X_val = joblib.load(splits_dir / "X_val.joblib")
print("Loaded successfully:", X_val.shape)

def load_data():
    """Load validation splits — same as lightgbm_model.py load_splits()"""
    print("Loading validation splits...")
    X_val = joblib.load(SPLITS_DIR / "X_val.joblib")
    y_val = joblib.load(SPLITS_DIR / "y_val.joblib")

    print(f"  X_val shape : {X_val.shape}")
    print(f"  Fraud rate  : {y_val.mean():.4f}")
    return X_val, y_val


def load_model():
    """Load saved LightGBM model"""
    model_path = MODELS_DIR / "lightgbm_model.joblib"
    print(f"\nLoading model from {model_path}...")
    model = joblib.load(model_path)
    print("  Model loaded successfully")
    return model


def generate_predictions(model, X_val, y_val):
    """
    Run model inference on validation set.
    Mirrors evaluate() in lightgbm_model.py but saves results instead of printing.
    """
    print("\nGenerating predictions...")

    # Keep card columns BEFORE dropping TransactionID for predictions
    # These are needed for dashboard user filtering
    card_cols = ["card1", "card2", "card3", "card4", "card5", "card6"]
    meta_cols = ["TransactionDT", "TransactionAmt", "ProductCD",
                 "P_emaildomain", "addr1"]

    # Save metadata before dropping columns for model input
    meta = X_val[[c for c in card_cols + meta_cols if c in X_val.columns]].copy()

    # Prepare features for model — drop TransactionID (same as training)
    X_model = X_val.copy()
    if "TransactionID" in X_model.columns:
        X_model = X_model.drop(columns=["TransactionID"])

    # Convert dtypes to float32 — matches load_splits() in lightgbm_model.py
    X_model = X_model.astype({col: "float32" for col in X_model.columns})

    # Run inference
    fraud_probability = model.predict_proba(X_model)[:, 1]
    model_decision    = (fraud_probability >= THRESHOLD).astype(int)

    print(f"  Predictions generated: {len(fraud_probability):,} transactions")
    print(f"  Flagged as fraud     : {model_decision.sum():,} "
          f"({model_decision.mean()*100:.2f}%)")

    return fraud_probability, model_decision, meta


def build_predictions_csv(X_val, y_val, fraud_probability, model_decision, meta):
    """
    Assemble final predictions DataFrame and save to CSV.
    This is what the dashboard will read.
    """
    print("\nBuilding predictions CSV...")

    predictions_df = meta.copy()
    predictions_df["isFraud"]           = y_val.values
    predictions_df["fraud_probability"] = fraud_probability
    predictions_df["model_decision"]    = model_decision

    # Add fake calendar date for dashboard display
    # TransactionDT is timedelta in seconds from a reference point
    BASE_DATE = pd.Timestamp("2020-01-01")
    if "TransactionDT" in predictions_df.columns:
        predictions_df["transaction_date"] = (
            BASE_DATE + pd.to_timedelta(predictions_df["TransactionDT"], unit="s")
        )

    # Add user/card labels for dashboard filtering
    predictions_df["demo_user"] = "Unknown"
    predictions_df["demo_card"] = "Unknown"

    for user, cards in DEMO_USERS.items():
        for card_label, card_info in cards.items():
            mask = predictions_df["card1"] == card_info["card1"]
            predictions_df.loc[mask, "demo_user"] = user
            predictions_df.loc[mask, "demo_card"] = card_label

    # Save to disk
    output_path = OUTPUT_DIR / "predictions.csv"
    predictions_df.to_csv(output_path, index=False)
    print(f"\n  Saved to: {output_path}")
    print(f"  Shape   : {predictions_df.shape}")
    print(f"  Columns : {predictions_df.columns.tolist()}")

    # Quick demo user transaction counts
    print("\nDemo user transaction counts:")
    summary = predictions_df[predictions_df["demo_user"] != "Unknown"].groupby(
        ["demo_user", "demo_card"]
    ).agg(
        transactions=("fraud_probability", "count"),
        fraud_flagged=("model_decision", "sum"),
        actual_fraud=("isFraud", "sum")
    )
    print(summary.to_string())

    return predictions_df


def main():
    print("=" * 55)
    print("FRAUD DETECTION — PREDICTION PIPELINE")
    print("=" * 55)

    X_val, y_val           = load_data()
    model                  = load_model()
    fraud_prob, decisions, meta = generate_predictions(model, X_val, y_val)
    predictions_df         = build_predictions_csv(
                                X_val, y_val, fraud_prob, decisions, meta
                             )

    print("\n" + "=" * 55)
    print("predictions.csv ready for dashboard!")
    print("=" * 55)

    return predictions_df


if __name__ == "__main__":
    main()