import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from src.config import *

class CategoricalEncoder:
    # Fits LabelEncoders on the TRAINING set only.
    # Call .fit() on train, then .transform() on both train and val/test.
    
    def __init__(self):
        self.encoders: dict = {}
        self.categorical_columns: list = []

    def fit(self, df: pd.DataFrame, target_col: str = "isFraud") -> "CategoricalEncoder":
        exclude = [target_col, "TransactionID"]
        self.categorical_columns = [
            col for col in df.select_dtypes(include=["object", "string", "category"]).columns
            if col not in exclude
        ]

        for col in self.categorical_columns:
            le = LabelEncoder()
            non_null = df[col].dropna().astype(str)
            if len(non_null) > 0:
                le.fit(non_null)
            self.encoders[col] = le

        print(f"[CategoricalEncoder] Fitted {len(self.categorical_columns)} categorical columns")
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df_encoded = df.copy()

        for col in self.categorical_columns:
            if col not in df_encoded.columns or col not in self.encoders:
                continue

            le = self.encoders[col]
            known = set(le.classes_)

            # Build a fully encoded series as int, unseen values → -1
            new_col = pd.Series(-1, index=df_encoded.index, dtype="int32")
            mask = df_encoded[col].notna()

            if mask.sum() > 0:
                values = df_encoded.loc[mask, col].astype(str)
                new_col[mask] = values.map(
                    lambda x: le.transform([x])[0] if x in known else -1
                ).astype("int32")

            df_encoded[col] = new_col

        return df_encoded

    def fit_transform(self, df: pd.DataFrame, target_col: str = "isFraud") -> pd.DataFrame:
        return self.fit(df, target_col).transform(df)

    def save(self, path=None):
        path = Path(path or PROCESSED_DIR / "models" / "encoder.joblib")
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        print(f"[CategoricalEncoder] Saved to {path}")
        return path

    @classmethod
    def load(cls, path=None) -> "CategoricalEncoder":
        path = Path(path or PROCESSED_DIR / "models" / "encoder.joblib")
        obj = joblib.load(path)
        print(f"[CategoricalEncoder] Loaded from {path}")
        return obj


def encode_categorical(df, output_file=None, target_col="isFraud"):
    # Legacy convenience function — fits and transforms on the same df.
    encoder = CategoricalEncoder()
    df_encoded = encoder.fit_transform(df, target_col=target_col)
    if output_file:
        df_encoded.to_csv(output_file, index=False)
    return df_encoded