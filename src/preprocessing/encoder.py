"""
Categorical encoding for fraud detection dataset using LabelEncoder.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from src.config import *


class CategoricalEncoder:
    def __init__(self):
        self.encoders = {}
        self.categorical_columns = []

    def fit(self, df, target_col='isFraud'):
        exclude = [target_col, 'TransactionID']
        self.categorical_columns = [
            col for col in df.select_dtypes(include=['object']).columns
            if col not in exclude
        ]

        for col in self.categorical_columns:
            le = LabelEncoder()
            non_null = df[col].dropna().astype(str)
            if len(non_null) > 0:
                le.fit(non_null)
            self.encoders[col] = le

        return self

    def transform(self, df):
        df_encoded = df.copy()

        for col in self.categorical_columns:
            if col not in df_encoded.columns or col not in self.encoders:
                continue

            le = self.encoders[col]
            known = set(le.classes_)

            # Build a fully encoded series as int, avoiding pyarrow dtype conflicts
            new_col = pd.Series(-1, index=df_encoded.index, dtype='int32')
            mask = df_encoded[col].notna()

            if mask.sum() > 0:
                values = df_encoded.loc[mask, col].astype(str)
                new_col[mask] = values.map(
                    lambda x: le.transform([x])[0] if x in known else -1
                ).astype('int32')

            df_encoded[col] = new_col

        return df_encoded

    def fit_transform(self, df, target_col='isFraud'):
        return self.fit(df, target_col).transform(df)


def encode_categorical(df, output_file=None, target_col='isFraud'):
    encoder = CategoricalEncoder()
    df_encoded = encoder.fit_transform(df, target_col=target_col)

    if output_file:
        df_encoded.to_csv(output_file, index=False)

    return df_encoded