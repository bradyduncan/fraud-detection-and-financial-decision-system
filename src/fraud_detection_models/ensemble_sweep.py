import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import (
    roc_auc_score,
    recall_score,
    precision_score,
    f1_score,
)

BASE_DIR   = Path(__file__).resolve().parents[2]
SPLITS_DIR = BASE_DIR / "data" / "processed" / "splits"
MODELS_DIR = BASE_DIR / "data" / "processed" / "models"

# Minimum precision floor — results below this are ignored
# Set to 0.0 if you want to see everything
MIN_PRECISION = 0.20

# Weight steps — each model gets weights from this list
# Must sum to 1.0 across all three models
WEIGHT_STEPS = [0.25, 0.30, 0.35, 0.40, 0.45, 0.50]

# Threshold values to test
THRESHOLD_STEPS = [round(x, 2) for x in np.arange(0.15, 0.55, 0.05)]

# How many top results to show
TOP_N = 15


def load_data():
    print("Loading validation splits...")
    X_val = joblib.load(SPLITS_DIR / "X_val.joblib")
    y_val = joblib.load(SPLITS_DIR / "y_val.joblib")

    if "TransactionID" in X_val.columns:
        X_val = X_val.drop(columns=["TransactionID"])

    X_val = X_val.astype({col: "float32" for col in X_val.columns})
    y_val = y_val.astype(int)

    print(f"  X_val shape : {X_val.shape}")
    print(f"  Fraud rate  : {y_val.mean():.4f}")
    return X_val, y_val


def load_models():
    print("\nLoading models...")
    model_files = {
        "xgboost":  MODELS_DIR / "xgboost.joblib",
        "lightgbm": MODELS_DIR / "lightgbm_model.joblib",
        "catboost": MODELS_DIR / "catboost.joblib",
    }
    models = {}
    for name, path in model_files.items():
        if not path.exists():
            raise FileNotFoundError(f"Model not found: {path}")
        models[name] = joblib.load(path)
        print(f"  Loaded {name}")
    return models


def get_individual_probas(models, X_val):
    print("\nGenerating individual model probabilities...")
    probas = {}
    for name, model in models.items():
        probas[name] = model.predict_proba(X_val)[:, 1]
        print(f"  {name:10s} → mean fraud prob: {probas[name].mean():.4f}")
    return probas


def generate_weight_combinations():
    """
    Generate all valid weight combinations where:
    - Each weight comes from WEIGHT_STEPS
    - All three weights sum to exactly 1.0
    """
    combos = []
    for xgb in WEIGHT_STEPS:
        for lgbm in WEIGHT_STEPS:
            cat = round(1.0 - xgb - lgbm, 2)
            if cat in WEIGHT_STEPS and cat > 0:
                combos.append((xgb, lgbm, cat))
    return combos


def sweep(probas, y_val):
    weight_combos = generate_weight_combinations()
    print(f"\nTesting {len(weight_combos)} weight combinations")
    print(f"Testing {len(THRESHOLD_STEPS)} threshold values")
    print(f"Total configurations: {len(weight_combos) * len(THRESHOLD_STEPS)}")
    print(f"Minimum precision floor: {MIN_PRECISION}")
    print("\nRunning sweep...")

    results = []

    for xgb_w, lgbm_w, cat_w in weight_combos:
        # Compute weighted ensemble probability
        ensemble_proba = (
            probas["xgboost"]  * xgb_w +
            probas["lightgbm"] * lgbm_w +
            probas["catboost"] * cat_w
        )

        auc = roc_auc_score(y_val, ensemble_proba)

        for threshold in THRESHOLD_STEPS:
            y_pred = (ensemble_proba >= threshold).astype(int)

            recall    = recall_score(y_val, y_pred, zero_division=0)
            precision = precision_score(y_val, y_pred, zero_division=0)
            f1        = f1_score(y_val, y_pred, zero_division=0)

            # Skip results below precision floor
            if precision < MIN_PRECISION:
                continue

            results.append({
                "xgb_weight":  xgb_w,
                "lgbm_weight": lgbm_w,
                "cat_weight":  cat_w,
                "threshold":   threshold,
                "auc":         round(auc, 4),
                "recall":      round(float(recall), 4),
                "precision":   round(float(precision), 4),
                "f1":          round(float(f1), 4),
            })

    return pd.DataFrame(results)


def print_results(df):
    if df.empty:
        print("\nNo results found above the minimum precision floor.")
        print(f"Try lowering MIN_PRECISION (currently {MIN_PRECISION})")
        return

    # Sort by recall descending, then AUC descending
    df_sorted = df.sort_values(
        ["recall", "auc"], ascending=[False, False]
    ).reset_index(drop=True)

    print("\n" + "=" * 75)
    print(f"TOP {TOP_N} CONFIGURATIONS BY RECALL (precision >= {MIN_PRECISION})")
    print("=" * 75)
    print(f"{'Rank':<5} {'XGB':>5} {'LGBM':>6} {'CAT':>5} {'Thresh':>7} "
          f"{'AUC':>7} {'Recall':>7} {'Prec':>7} {'F1':>7}")
    print("-" * 75)

    for i, row in df_sorted.head(TOP_N).iterrows():
        print(f"{i+1:<5} {row['xgb_weight']:>5.2f} {row['lgbm_weight']:>6.2f} "
              f"{row['cat_weight']:>5.2f} {row['threshold']:>7.2f} "
              f"{row['auc']:>7.4f} {row['recall']:>7.4f} "
              f"{row['precision']:>7.4f} {row['f1']:>7.4f}")

    print("\n" + "=" * 75)
    print("BEST CONFIGURATION BY RECALL:")
    print("=" * 75)
    best = df_sorted.iloc[0]
    print(f"  Weights    : XGBoost={best['xgb_weight']:.0%}  "
          f"LightGBM={best['lgbm_weight']:.0%}  "
          f"CatBoost={best['cat_weight']:.0%}")
    print(f"  Threshold  : {best['threshold']:.4f}")
    print(f"  ROC-AUC    : {best['auc']:.4f}")
    print(f"  Recall     : {best['recall']:.4f}")
    print(f"  Precision  : {best['precision']:.4f}")
    print(f"  F1 Score   : {best['f1']:.4f}")

    # Also show best balanced config (highest F1)
    print("\n" + "=" * 75)
    print("BEST CONFIGURATION BY F1 (most balanced):")
    print("=" * 75)
    best_f1 = df_sorted.sort_values("f1", ascending=False).iloc[0]
    print(f"  Weights    : XGBoost={best_f1['xgb_weight']:.0%}  "
          f"LightGBM={best_f1['lgbm_weight']:.0%}  "
          f"CatBoost={best_f1['cat_weight']:.0%}") 
    print(f"  Threshold  : {best_f1['threshold']:.4f}")
    print(f"  ROC-AUC    : {best_f1['auc']:.4f}")
    print(f"  Recall     : {best_f1['recall']:.4f}")
    print(f"  Precision  : {best_f1['precision']:.4f}")
    print(f"  F1 Score   : {best_f1['f1']:.4f}")

    # Save full results to CSV
    output_path = BASE_DIR / "data" / "processed" / "ensemble_sweep_results.csv"
    df_sorted.to_csv(output_path, index=False)
    print(f"\n  Full results saved to: {output_path}")


def main():
    print("=" * 75)
    print("ENSEMBLE WEIGHT + THRESHOLD SWEEP")
    print("Goal: find configuration with highest recall (precision >= {})".format(
        MIN_PRECISION))
    print("=" * 75)

    X_val, y_val  = load_data()
    models        = load_models()
    probas        = get_individual_probas(models, X_val)
    results_df    = sweep(probas, y_val)
    print_results(results_df)

    return results_df


if __name__ == "__main__":
    main()