import pandas as pd
from pathlib import Path
from src.config import *

def _read_csv_arrow(path: str) -> pd.DataFrame:
    # Read CSV with pyarrow-backed dtypes to reduce peak memory usage.

    try:
        return pd.read_csv(path, engine="pyarrow", dtype_backend="pyarrow")
    except Exception:
        return pd.read_csv(path, low_memory=False)

def merge_transaction_identity(transaction_file, identity_file, output_file=None):    
    # Load transaction data
    print("Loading transaction data")
    transaction_df = _read_csv_arrow(transaction_file)
    
    # Load identity data
    print("Loading identity data")
    identity_df = _read_csv_arrow(identity_file)
    
    # Analyze coverage
    print("Analyzing identity coverage")
    total_transactions = len(transaction_df)
    transactions_with_identity = transaction_df['TransactionID'].isin(identity_df['TransactionID']).sum()
    coverage_pct = (transactions_with_identity / total_transactions) * 100
    
    print(f"\nIdentity Coverage:")
    print(f"Total transactions: {total_transactions:,}")
    print(f"Transactions with identity: {transactions_with_identity:,}")
    print(f"Coverage: {coverage_pct:.2f}%")
    
    # Add has_identity flag
    print(4, "Creating 'has_identity' feature")
    transaction_df = transaction_df.copy()
    transaction_df['has_identity'] = transaction_df['TransactionID'].isin(
        identity_df['TransactionID']
    ).astype(int)
    print(f"Added 'has_identity' feature (1={transactions_with_identity:,}, 0={total_transactions-transactions_with_identity:,})")
    
    # Merge
    print("Merging datasets (left join)")
    merged_df = transaction_df.merge(
        identity_df,
        on='TransactionID',
        how='left',
        validate='one_to_one'
    )
    
    print(f"\nMerged Data:")
    print(f"Shape: {merged_df.shape}")
    print(f"Rows: {merged_df.shape[0]:,}")
    print(f"Columns: {merged_df.shape[1]}")
    
    # Verify
    assert len(merged_df) == len(transaction_df), "Lost transactions during merge!"
    print("No transactions lost during merge")
    
    # Check target distribution
    if 'isFraud' in merged_df.columns:
        fraud_count = merged_df['isFraud'].sum()
        fraud_pct = (fraud_count / len(merged_df)) * 100
        print(f"\nTarget Distribution:")
        print(f"  Legitimate: {len(merged_df) - fraud_count:,} ({100-fraud_pct:.2f}%)")
        print(f"  Fraud: {fraud_count:,} ({fraud_pct:.2f}%)")
    
    # Save if output file specified
    if output_file:
        print("Saving merged data")
        merged_df.to_csv(output_file, index=False)

    print("Data merging complete!")    
    return merged_df

def main():
    merged_df = merge_transaction_identity(
        TRAIN_TRANSACTION_FILE,
        TRAIN_IDENTITY_FILE,
        MERGED_DATA_FILE
    )
    return merged_df

if __name__ == "__main__":
    main()
