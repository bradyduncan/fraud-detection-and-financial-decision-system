# generate_shap.py — Pre-compute SHAP values for dashboard explainability
# Run ONCE before launching the dashboard:
#   python -m src.fraud_detection_models.generate_shap

import joblib
import numpy as np
import pandas as pd
import shap
from pathlib import Path

BASE_DIR   = Path(__file__).resolve().parents[2]
SPLITS_DIR = BASE_DIR / "data" / "processed" / "splits"
MODELS_DIR = BASE_DIR / "data" / "processed" / "models"
OUTPUT_DIR = BASE_DIR / "data" / "processed"

# 
DEMO_USERS = {
    "Alice": {"Card 1": 7919,  "Card 2": 15066},
    "Bob":   {"Card 1": 9500,  "Card 2": 6019},
    "Carol": {"Card 1": 15885, "Card 2": 7585},
    "Dave":  {"Card 1": 9633,  "Card 2": 12695},
}

# Number of transactions to compute SHAP for per card
# Keep this small — SHAP is slow on large datasets
# 200 per card = ~1200 total rows = manageable runtime (~2-3 mins)
SAMPLE_SIZE = 200

def load_data():
    print("Loading validation splits...")
    X_val = joblib.load(SPLITS_DIR / "X_val.joblib")
    y_val = joblib.load(SPLITS_DIR / "y_val.joblib")

    if "TransactionID" in X_val.columns:
        X_val = X_val.drop(columns=["TransactionID"])

    X_val = X_val.astype({col: "float32" for col in X_val.columns})
    y_val = y_val.astype(int)

    print(f"  X_val shape : {X_val.shape}")
    return X_val, y_val

def load_model():
    print("\nLoading LightGBM model...")
    model = joblib.load(MODELS_DIR / "lightgbm_model.joblib")
    print("  Model loaded successfully")
    return model


def sample_per_card(X_val, y_val) -> pd.DataFrame:
    print(f"\nSampling {SAMPLE_SIZE} transactions per card...")
    samples = []

    for user, cards in DEMO_USERS.items():
        for card_label, card1_val in cards.items():
            # Get all transactions for this card
            card_mask = X_val["card1"] == card1_val
            X_card    = X_val[card_mask].copy()
            y_card    = y_val[card_mask].copy()

            if len(X_card) == 0:
                print(f"  WARNING: No transactions found for {user} {card_label}")
                continue

            # Try to get a mix of fraud + legit transactions
            # This makes SHAP explanations more informative
            fraud_idx = y_card[y_card == 1].index
            legit_idx = y_card[y_card == 0].index

            n_fraud = min(len(fraud_idx), SAMPLE_SIZE // 4)  # 25% fraud
            n_legit = SAMPLE_SIZE - n_fraud                   # 75% legit

            sampled_idx = pd.Index([])
            if n_fraud > 0:
                sampled_idx = sampled_idx.append(
                    pd.Index(np.random.choice(fraud_idx, n_fraud, replace=False))
                )
            if n_legit > 0:
                sampled_idx = sampled_idx.append(
                    pd.Index(np.random.choice(
                        legit_idx,
                        min(n_legit, len(legit_idx)),
                        replace=False
                    ))
                )

            X_sample = X_card.loc[sampled_idx].copy()
            X_sample["demo_user"] = user
            X_sample["demo_card"] = card_label
            X_sample["isFraud"]   = y_card.loc[sampled_idx].values
            samples.append(X_sample)

            print(f"  {user} {card_label}: {len(X_sample)} rows "
                  f"({n_fraud} fraud, {len(X_sample)-n_fraud} legit)")

    combined = pd.concat(samples, ignore_index=True)
    print(f"\n  Total sampled: {len(combined)} rows")
    return combined


def compute_shap(model, sampled_df):
    print("\nComputing SHAP values (this may take 2-3 minutes)...")

    # Separate feature columns from metadata columns
    meta_cols    = ["demo_user", "demo_card", "isFraud"]
    feature_cols = [c for c in sampled_df.columns if c not in meta_cols]
    X_features   = sampled_df[feature_cols]

    # TreeExplainer is optimised for tree-based models like LightGBM
    # Much faster than the generic KernelExplainer
    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_features)

    # LightGBM binary classifier returns list [class0, class1]
    # We want class 1 (fraud) SHAP values
    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    # Build output DataFrame
    shap_df = pd.DataFrame(shap_values, columns=feature_cols)

    # Add metadata columns back
    shap_df["demo_user"] = sampled_df["demo_user"].values
    shap_df["demo_card"] = sampled_df["demo_card"].values
    shap_df["isFraud"]   = sampled_df["isFraud"].values

    # Add original feature values (needed for SHAP dependence plots)
    for col in ["TransactionAmt", "card1", "card4", "card6"]:
        if col in feature_cols:
            shap_df[f"val_{col}"] = sampled_df[col].values

    print(f"  SHAP values computed: {shap_df.shape}")
    return shap_df


def save_outputs(shap_df, sampled_df, feature_cols):
    # Save full SHAP values
    shap_path = OUTPUT_DIR / "shap_values.csv"
    shap_df.to_csv(shap_path, index=False)
    print(f"\n  Saved shap_values.csv → {shap_path}")

    # Compute global feature importance
    # Mean absolute SHAP = average impact of each feature across all transactions
    meta_cols  = ["demo_user", "demo_card", "isFraud"] + \
                 [c for c in shap_df.columns if c.startswith("val_")]
    shap_only  = shap_df.drop(columns=meta_cols, errors="ignore")

    importance_df = pd.DataFrame({
        "feature":    shap_only.columns,
        "importance": shap_only.abs().mean().values
    }).sort_values("importance", ascending=False).reset_index(drop=True)

    importance_path = OUTPUT_DIR / "shap_feature_importance.csv"
    importance_df.to_csv(importance_path, index=False)
    print(f"  Saved shap_feature_importance.csv → {importance_path}")
    print(f"\n  Top 10 most important features:")
    print(importance_df.head(10).to_string(index=False))

    return importance_df


def main():
    print("BASE_DIR:", BASE_DIR)
    print("SPLITS_DIR:", SPLITS_DIR)
    print("File exists:", (SPLITS_DIR / "X_val.joblib").exists())
    print("=" * 55)
    print("SHAP VALUE GENERATION PIPELINE")
    print("=" * 55)

    np.random.seed(42)  # reproducibility

    X_val, y_val   = load_data()
    model          = load_model()
    sampled_df     = sample_per_card(X_val, y_val)

    feature_cols   = [c for c in sampled_df.columns
                      if c not in ["demo_user", "demo_card", "isFraud"]]

    shap_df        = compute_shap(model, sampled_df)
    importance_df  = save_outputs(shap_df, sampled_df, feature_cols)

    print("SHAP generation complete! Files saved:")
    print("  → data/processed/shap_values.csv")
    print("  → data/processed/shap_feature_importance.csv")
    print("These files are now ready for the dashboard.")

    return shap_df, importance_df


if __name__ == "__main__":
    main()