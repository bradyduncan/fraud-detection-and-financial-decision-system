# Feature engineering for the temporal features 
# src/temporal_features.py

import numpy as np
import pandas as pd

def _as_time_column_from_transaction_dt(series: pd.Series) -> pd.Series:
    # Convert TransactionDT-style seconds-since-origin into a pandas datetime column.
    # We don't know the true origin; that's fine because we only use *relative ordering* and time deltas / rolling windows.

    # Origin doesn't matter, just order
    secs = pd.to_numeric(series, errors="coerce")
    return pd.Timestamp("1970-01-01") + pd.to_timedelta(secs, unit="s")


def _window_label(seconds: int) -> str:
    if seconds % 86400 == 0: # 1 day
        return f"{seconds // 86400}d"
    if seconds % 3600 == 0: # 1 hour
        return f"{seconds // 3600}h"
    if seconds % 60 == 0: # 1 minute
        return f"{seconds // 60}m"
    return f"{seconds}s"


def add_temporal_features(
    df: pd.DataFrame,
    time_col: str = "TransactionDT",
    amount_col: str = "TransactionAmt",
    transaction_id_col: str = "TransactionID",
    entity_defs: dict | None = None,
    windows_seconds: list[int] | None = None,
    add_amount_rolling: bool = True,
    add_velocity: bool = True,
    inplace: bool = False,
) -> pd.DataFrame:
    print("\nAdding temporal features")
    if not inplace:
        df = df.copy(deep=False)

    if time_col not in df.columns:
        raise KeyError(f"Expected time_col='{time_col}' in dataframe columns.")
    if amount_col not in df.columns:
        add_amount_rolling = False

    # Default entity definitions
    if entity_defs is None:
        entity_defs = {
            "card1": ["card1"],
            "card1_addr1": ["card1", "addr1"],
            "card1_card2": ["card1", "card2"],
            "email_p": ["P_emaildomain"],
        }

    if windows_seconds is None:
        windows_seconds = [3600, 86400, 7 * 86400]  # 1h, 24h, 1w

    # Create internal datetime column for rolling-window ops
    ts_col = "_ts_dt_internal"
    df[ts_col] = _as_time_column_from_transaction_dt(df[time_col])

    # Sort by time, TransactionID
    sort_cols = [ts_col]
    if transaction_id_col in df.columns:
        sort_cols.append(transaction_id_col)

    df = df.sort_values(sort_cols, kind="mergesort").reset_index(drop=True)

    # Ensure windows are unique and sorted
    windows_seconds = sorted(set(int(w) for w in windows_seconds if w is not None and int(w) > 0))

    # Core features per entity
    for entity_name, cols in entity_defs.items():
        # Skip entities whose columns are not present
        missing_cols = [c for c in cols if c not in df.columns]
        if missing_cols:
            continue

        # group by columns
        g = df.groupby(cols, sort=False, observed=True)

        # Time since last transaction (seconds)
        dt = g[ts_col].diff()
        feat_tsl = f"tf_{entity_name}_time_since_last_s"
        df[feat_tsl] = dt.dt.total_seconds()

        # First-transaction indicator
        df[f"tf_{entity_name}_is_first_txn"] = df[feat_tsl].isna().astype("int8")

        # Count and amount stats for rolling features
        for w in windows_seconds:
            label = _window_label(w)
            window_str = f"{w}s"

            # Rolling count of prior transactions in the window; excludes current timestamp - no leakage.
            roll_count = np.full(len(df), np.nan, dtype="float32")
            for _, gdf in g:
                roll = gdf.rolling(window=window_str, on=ts_col, closed="left")[ts_col].count()
                roll_count[gdf.index] = roll.to_numpy(dtype="float32")
            df[f"tf_{entity_name}_txn_count_{label}"] = roll_count

            if add_amount_rolling and amount_col in df.columns:
                # Rolling amount stats (prior only)
                amt_mean = np.full(len(df), np.nan, dtype="float32")
                amt_std = np.full(len(df), np.nan, dtype="float32")
                amt_sum = np.full(len(df), np.nan, dtype="float32")
                for _, gdf in g:
                    roll_amt = gdf.rolling(window=window_str, on=ts_col, closed="left")[amount_col]
                    amt_mean[gdf.index] = roll_amt.mean().to_numpy(dtype="float32")
                    amt_std[gdf.index] = roll_amt.std().to_numpy(dtype="float32")
                    amt_sum[gdf.index] = roll_amt.sum().to_numpy(dtype="float32")

                df[f"tf_{entity_name}_amt_mean_{label}"] = amt_mean
                df[f"tf_{entity_name}_amt_std_{label}"] = amt_std
                df[f"tf_{entity_name}_amt_sum_{label}"] = amt_sum

                if add_velocity:
                    # Sum per hour/day-equivalent (normalized to per-second)
                    denom = float(w) if w > 0 else 1.0
                    df[f"tf_{entity_name}_amt_velocity_{label}"] = (
                        df[f"tf_{entity_name}_amt_sum_{label}"] / denom
                    ).astype("float32")

    # Global temporal features
    # Hour-of-day / day-of-week proxies are meaningful even with arbitrary origin
    df["tf_hour"] = df[ts_col].dt.hour.astype("int8")
    df["tf_dayofweek"] = df[ts_col].dt.dayofweek.astype("int8")

    # Clean up internal column
    df = df.drop(columns=[ts_col])
    
    print("\nTemporal Feature Engineering Completed")

    return df