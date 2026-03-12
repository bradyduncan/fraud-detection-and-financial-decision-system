import joblib
import pandas as pd
import numpy as np
from pathlib import Path

SPLITS_DIR = Path('data/processed/splits')

print("Loading with current pandas...")
X_train = joblib.load(SPLITS_DIR / 'X_train.joblib')
y_train = joblib.load(SPLITS_DIR / 'y_train.joblib')
X_val   = joblib.load(SPLITS_DIR / 'X_val.joblib')
y_val   = joblib.load(SPLITS_DIR / 'y_val.joblib')

print("Converting to clean dtypes...")
# Convert everything to plain numpy-backed types
X_train = X_train.astype({col: "float32" for col in X_train.columns})
X_val   = X_val.astype({col: "float32" for col in X_val.columns})
y_train = y_train.astype(int)
y_val   = y_val.astype(int)

print("Re-saving clean versions...")
joblib.dump(X_train, SPLITS_DIR / 'X_train.joblib')
joblib.dump(y_train, SPLITS_DIR / 'y_train.joblib')
joblib.dump(X_val,   SPLITS_DIR / 'X_val.joblib')
joblib.dump(y_val,   SPLITS_DIR / 'y_val.joblib')

print("Done! All splits re-saved with clean dtypes.")
print(f"X_val shape : {X_val.shape}")
print(f"X_val dtypes: {X_val.dtypes.unique()}")