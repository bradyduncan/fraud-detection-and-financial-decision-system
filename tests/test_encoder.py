"""
Unit tests for src/preprocessing/encoder.py

Tests cover:
  - CategoricalEncoder: fit/transform correctness, unseen-value handling,
    numeric passthrough, and protection of isFraud / TransactionID.
"""

import pandas as pd
import numpy as np
import pytest

from src.preprocessing.encoder import CategoricalEncoder


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_df(**kwargs) -> pd.DataFrame:
    """Build a small DataFrame from keyword-argument column arrays."""
    return pd.DataFrame(kwargs)


# ---------------------------------------------------------------------------
# CategoricalEncoder tests
# ---------------------------------------------------------------------------

class TestCategoricalEncoder:

    def test_fit_transform_converts_strings_to_integers(self):
        """String/object columns must be encoded as non-negative integers after fit_transform."""
        data = _make_df(
            TransactionID=range(4),
            isFraud=[0, 1, 0, 1],
            card_type=["visa", "mastercard", "visa", "amex"],
        )
        encoder = CategoricalEncoder()
        result = encoder.fit_transform(data)
        assert result["card_type"].dtype in [np.dtype("int32"), np.dtype("int64"), "int32", "int64"]
        assert result["card_type"].min() >= 0

    def test_unseen_category_maps_to_minus_one(self):
        """Categories in validation that were not in training must map to -1 without crashing."""
        train = _make_df(
            TransactionID=range(3),
            isFraud=[0, 0, 1],
            card_type=["visa", "mastercard", "visa"],
        )
        val = _make_df(
            TransactionID=range(3, 6),
            isFraud=[0, 1, 0],
            card_type=["visa", "amex", "discover"],  # "amex" and "discover" are unseen
        )
        encoder = CategoricalEncoder()
        encoder.fit(train)
        result = encoder.transform(val)
        # "amex" and "discover" must become -1
        assert result["card_type"].iloc[1] == -1
        assert result["card_type"].iloc[2] == -1

    def test_numeric_columns_are_left_unchanged(self):
        """Purely numeric columns must pass through the encoder without modification."""
        data = _make_df(
            TransactionID=range(4),
            isFraud=[0, 1, 0, 1],
            TransactionAmt=[10.5, 20.0, 15.3, 8.9],
            cat_col=["a", "b", "a", "c"],
        )
        encoder = CategoricalEncoder()
        result = encoder.fit_transform(data)
        pd.testing.assert_series_equal(
            result["TransactionAmt"],
            data["TransactionAmt"],
            check_names=True,
        )

    def test_isfraud_and_transaction_id_are_never_encoded(self):
        """isFraud and TransactionID must not appear in the encoder's categorical columns list."""
        data = _make_df(
            TransactionID=range(4),
            isFraud=[0, 1, 0, 1],
            card_type=["visa", "mastercard", "visa", "amex"],
        )
        encoder = CategoricalEncoder()
        encoder.fit(data)
        assert "isFraud" not in encoder.categorical_columns
        assert "TransactionID" not in encoder.categorical_columns

    def test_known_category_encoding_is_consistent(self):
        """The same category must receive the same integer in both fit_transform and transform."""
        train = _make_df(
            TransactionID=range(3),
            isFraud=[0, 0, 1],
            card_type=["visa", "mastercard", "visa"],
        )
        val = _make_df(
            TransactionID=range(3, 5),
            isFraud=[1, 0],
            card_type=["mastercard", "visa"],
        )
        encoder = CategoricalEncoder()
        train_enc = encoder.fit_transform(train)
        val_enc = encoder.transform(val)

        visa_code_train = train_enc.loc[train_enc["card_type"] == train_enc["card_type"].iloc[0], "card_type"].iloc[0]
        visa_code_val = val_enc["card_type"].iloc[1]   # val row 1 is "visa"
        assert visa_code_train == visa_code_val
