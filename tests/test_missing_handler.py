"""
Unit tests for src/preprocessing/missing_handler.py

Tests cover:
  - drop_extreme_missing: column dropping logic and CRITICAL_FEATURES protection
  - MissingValueImputer: fit/transform correctness, sentinel logic, and guard clauses
"""

import numpy as np
import pandas as pd
import pytest

from src.preprocessing.missing_handler import MissingValueImputer, drop_extreme_missing


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_df(**kwargs) -> pd.DataFrame:
    """Build a small DataFrame from keyword-argument column arrays."""
    return pd.DataFrame(kwargs)


# ---------------------------------------------------------------------------
# drop_extreme_missing
# ---------------------------------------------------------------------------

class TestDropExtremeMissing:

    def test_drops_column_above_threshold(self):
        """Columns where >90% of values are NaN must be dropped."""
        data = _make_df(
            TransactionID=range(10),
            isFraud=[0] * 10,
            bad_col=[np.nan] * 10,          # 100% missing
            good_col=list(range(10)),
        )
        result, dropped = drop_extreme_missing(data, threshold=90)
        assert "bad_col" in dropped
        assert "bad_col" not in result.columns

    def test_keeps_column_below_threshold(self):
        """Columns with <=90% missing values must be kept."""
        data = _make_df(
            TransactionID=range(10),
            isFraud=[0] * 10,
            ok_col=[np.nan, np.nan, 1, 2, 3, 4, 5, 6, 7, 8],  # 20% missing
        )
        result, dropped = drop_extreme_missing(data, threshold=90)
        assert "ok_col" not in dropped
        assert "ok_col" in result.columns

    def test_never_drops_critical_features(self):
        """CRITICAL_FEATURES (e.g. isFraud, TransactionAmt) must survive even if 100% missing."""
        data = _make_df(
            TransactionID=range(10),
            isFraud=[np.nan] * 10,           # 100% missing but critical
            TransactionAmt=[np.nan] * 10,    # 100% missing but critical
            expendable=[np.nan] * 10,
        )
        result, dropped = drop_extreme_missing(data, threshold=90)
        assert "isFraud" not in dropped
        assert "TransactionAmt" not in dropped
        assert "isFraud" in result.columns
        assert "TransactionAmt" in result.columns

    def test_returns_correct_dropped_list(self):
        """The second return value must list exactly the columns that were dropped."""
        data = _make_df(
            TransactionID=range(10),
            isFraud=[0] * 10,
            col_a=[np.nan] * 10,
            col_b=[np.nan] * 10,
            col_c=list(range(10)),
        )
        _, dropped = drop_extreme_missing(data, threshold=90)
        assert set(dropped) == {"col_a", "col_b"}


# ---------------------------------------------------------------------------
# MissingValueImputer
# ---------------------------------------------------------------------------

class TestMissingValueImputer:

    def test_fit_transform_leaves_zero_missing(self):
        """After fit_transform, the DataFrame must contain no NaN values."""
        data = _make_df(
            TransactionID=range(10),
            isFraud=[0] * 10,
            num_col=[1.0, np.nan, 3.0, np.nan, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
            cat_col=["a", None, "b", "a", None, "b", "a", "b", "a", "b"],
        )
        imputer = MissingValueImputer()
        result = imputer.fit_transform(data)
        assert result.isnull().sum().sum() == 0

    def test_transform_before_fit_raises_runtime_error(self):
        """Calling transform() on an unfitted imputer must raise RuntimeError."""
        data = _make_df(num_col=[1.0, np.nan])
        imputer = MissingValueImputer()
        with pytest.raises(RuntimeError):
            imputer.transform(data)

    def test_v_features_high_missing_get_sentinel(self):
        """V-features with >50% missing values must be filled with -999 (sentinel)."""
        # 8 out of 10 rows (80%) are NaN → sentinel
        v_col = [np.nan] * 8 + [1.0, 2.0]
        data = _make_df(
            TransactionID=range(10),
            isFraud=[0] * 10,
            V1=v_col,
        )
        imputer = MissingValueImputer()
        result = imputer.fit_transform(data)
        # All originally-NaN positions must now be -999
        assert (result["V1"][:8] == -999).all()

    def test_v_features_low_missing_get_median(self):
        """V-features with <=50% missing values must be filled with the median."""
        # Only 2 out of 10 (20%) are NaN → median
        v_col = [np.nan, np.nan, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0]
        data = _make_df(
            TransactionID=range(10),
            isFraud=[0] * 10,
            V2=v_col,
        )
        imputer = MissingValueImputer()
        result = imputer.fit_transform(data)
        expected_median = pd.Series(v_col).median()
        assert result["V2"].iloc[0] == pytest.approx(expected_median)

    def test_fit_on_train_transform_on_val(self):
        """Fill values learned from train must be applied unchanged to validation data."""
        train = _make_df(
            TransactionID=range(5),
            isFraud=[0] * 5,
            num_col=[10.0, 20.0, 30.0, np.nan, 40.0],
        )
        val = _make_df(
            TransactionID=range(5, 10),
            isFraud=[0] * 5,
            num_col=[np.nan, 50.0, 60.0, np.nan, 70.0],
        )
        imputer = MissingValueImputer()
        imputer.fit(train)
        result_val = imputer.transform(val)
        # Fill value learned from train (median of [10,20,30,40] = 25.0)
        assert result_val["num_col"].isnull().sum() == 0
        assert result_val["num_col"].iloc[0] == pytest.approx(25.0)
