"""
Feature selection module for IEEE-CIS Fraud Detection preprocessing pipeline.
Applies three sequential filters, preserving 'isFraud' and 'TransactionID' by default.
"""

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif

PROTECTED_COLS = ["isFraud", "TransactionID"]


def _get_protected(df: pd.DataFrame) -> list:
    """Return protected columns that actually exist in the dataframe."""
    return [c for c in PROTECTED_COLS if c in df.columns]


def _near_zero_variance_filter(
    df: pd.DataFrame,
    dominant_freq_threshold: float = 0.98,
    nunique_min: int = 2,
) -> pd.DataFrame:
    """
    Remove features with near-zero variance.

    Args:
        df: Input dataframe (fully numeric, post-encoding).
        dominant_freq_threshold: Max allowed frequency ratio for the most common value (default .98).
        nunique_min: Minimum number of unique values required.

    Returns:
        Filtered dataframe.
    """
    protected = _get_protected(df)
    drop_cols = []

    for col in df.columns:
        if col in protected:
            continue

        n_unique = df[col].nunique(dropna=True)
        if n_unique < nunique_min:
            drop_cols.append(col)
            continue

        # Check dominant frequency ratio
        mode_freq = df[col].value_counts(normalize=True, dropna=True)
        if len(mode_freq) > 0 and mode_freq.iloc[0] >= dominant_freq_threshold:
            drop_cols.append(col)

    print(
        f"[Near-Zero Variance] Dropping {len(drop_cols)} columns "
        f"(threshold={dominant_freq_threshold}, nunique_min={nunique_min})"
    )
    if drop_cols:
        print(f"  Dropped: {drop_cols[:20]}{'...' if len(drop_cols) > 20 else ''}")

    return df.drop(columns=drop_cols)


def _correlation_filter(
    df: pd.DataFrame,
    target_col: str = "isFraud",
    corr_threshold: float = 0.95,
) -> pd.DataFrame:
    """
    Remove redundant features using pairwise Pearson correlation.

    Args:
        df: Input dataframe (fully numeric).
        target_col: Name of the binary target column.
        corr_threshold: Correlation threshold above which one feature in a pair is dropped (default .95).

    Returns:
        Filtered dataframe.
    """
    protected = _get_protected(df)
    feature_cols = [c for c in df.columns if c not in protected]

    print(
        f"[Correlation Filter] Computing correlation matrix for "
        f"{len(feature_cols)} features (threshold={corr_threshold})..."
    )

    # Compute correlation matrix only for feature columns
    corr_matrix = df[feature_cols].corr().abs()

    # Compute each feature's absolute correlation with the target
    target_corr = df[feature_cols].corrwith(df[target_col]).abs()
    target_corr = target_corr.fillna(0)

    # Identify pairs to evaluate using upper triangle
    upper_tri = np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    upper_corr = corr_matrix.where(upper_tri)

    drop_cols = set()
    for col in feature_cols:
        if col in drop_cols:
            continue
        # Find features highly correlated with this column
        correlated = upper_corr[col][upper_corr[col] > corr_threshold].index.tolist()
        for other in correlated:
            if other in drop_cols:
                continue
            # Drop whichever has lower correlation with target
            if target_corr.get(col, 0) >= target_corr.get(other, 0):
                drop_cols.add(other)
            else:
                drop_cols.add(col)
                break

    print(f"[Correlation Filter] Dropping {len(drop_cols)} redundant columns")
    if drop_cols:
        sample = list(drop_cols)[:20]
        print(f"  Dropped: {sample}{'...' if len(drop_cols) > 20 else ''}")

    return df.drop(columns=list(drop_cols))


def _mutual_information_filter(
    df: pd.DataFrame,
    target_col: str = "isFraud",
    mi_threshold: float = 0.0,
    n_neighbors: int = 5,
    random_state: int = 42,
    sample_size: int = 50_000,
) -> pd.DataFrame:
    """
    Remove features with negligible mutual information with the target.

    Args:
        df: Input dataframe (fully numeric).
        target_col: Name of the binary target column.
        mi_threshold: Minimum MI score to keep a feature. Default 0.0
            drops only features with effectively zero information.
        n_neighbors: Number of neighbors for MI estimation (sklearn param).
        random_state: Random seed for reproducibility.
        sample_size: Max rows to use for MI estimation (for speed).

    Returns:
        Filtered dataframe.
    """
    protected = _get_protected(df)
    feature_cols = [c for c in df.columns if c not in protected]

    # Subsample for speed on large datasets
    if len(df) > sample_size:
        print(
            f"[Mutual Information] Subsampling {sample_size} rows "
            f"from {len(df)} for MI estimation"
        )
        df_sample = df.sample(n=sample_size, random_state=random_state)
    else:
        df_sample = df

    X = df_sample[feature_cols]
    y = df_sample[target_col]

    print(
        f"[Mutual Information] Computing MI scores for {len(feature_cols)} features..."
    )

    mi_scores = mutual_info_classif(
        X,
        y,
        discrete_features=False,
        n_neighbors=n_neighbors,
        random_state=random_state,
    )

    mi_series = pd.Series(mi_scores, index=feature_cols)
    drop_cols = mi_series[mi_series <= mi_threshold].index.tolist()

    print(
        f"[Mutual Information] Dropping {len(drop_cols)} features "
        f"with MI <= {mi_threshold}"
    )
    if drop_cols:
        print(f"  Dropped: {drop_cols[:20]}{'...' if len(drop_cols) > 20 else ''}")

    # Log summary statistics
    print(
        f"  MI score range: [{mi_series.min():.6f}, {mi_series.max():.6f}], "
        f"median={mi_series.median():.6f}"
    )

    return df.drop(columns=drop_cols)


def select_features(
    df: pd.DataFrame,
    target_col: str = "isFraud",
    nzv_dominant_freq: float = 0.98,
    corr_threshold: float = 0.95,
    mi_threshold: float = 0.0,
    mi_sample_size: int = 50_000,
    output_file=None,
) -> pd.DataFrame:
    """
    Run the full feature selection pipeline.

    Sequential filters applied:
        1. Near-zero variance removal
        2. Target-aware pairwise correlation removal
        3. Mutual information filter

    Args:
        df: Fully numeric dataframe (post-encoding, post-imputation).
        target_col: Binary target column name.
        nzv_dominant_freq: Threshold for near-zero variance filter.
        corr_threshold: Pearson correlation threshold for redundancy.
        mi_threshold: Minimum mutual information score to retain feature.
        mi_sample_size: Subsample size for MI computation.
        output_file: If provided, save result to this path.

    Returns:
        Filtered dataframe ready for modeling.
    """
    initial_cols = len(df.columns)
    print(f"Starting feature selection with {initial_cols} columns, {len(df)} rows")

    # Step 1: Near-zero variance
    df = _near_zero_variance_filter(
        df, dominant_freq_threshold=nzv_dominant_freq
    )
    after_nzv = len(df.columns)

    # Step 2: Pairwise correlation (target-aware)
    df = _correlation_filter(
        df, target_col=target_col, corr_threshold=corr_threshold
    )
    after_corr = len(df.columns)

    # Step 3: Mutual information
    df = _mutual_information_filter(
        df,
        target_col=target_col,
        mi_threshold=mi_threshold,
        sample_size=mi_sample_size,
    )
    after_mi = len(df.columns)

    # Summary
    print()
    print("Feature Selection Summary:")
    print(f"  Initial features:           {initial_cols}")
    print(f"  After near-zero variance:   {after_nzv} (-{initial_cols - after_nzv})")
    print(f"  After correlation filter:   {after_corr} (-{after_nzv - after_corr})")
    print(f"  After mutual information:   {after_mi} (-{after_corr - after_mi})")
    print(f"  Total removed:              {initial_cols - after_mi}")

    if output_file is not None:
        df.to_csv(output_file, index=False)
        print(f"Saved to {output_file}")

    return df