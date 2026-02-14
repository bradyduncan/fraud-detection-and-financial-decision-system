from pathlib import Path
import pandas as pd

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / 'data'
PROCESSED_DIR = DATA_DIR / 'processed'
FEATURE_ANALYSIS_DIR = PROCESSED_DIR / 'feature_analysis'

# Create directories
PROCESSED_DIR.mkdir(exist_ok=True)
FEATURE_ANALYSIS_DIR.mkdir(exist_ok=True)

# Input files
TRAIN_TRANSACTION_FILE = DATA_DIR / 'train_transaction.csv'
TRAIN_IDENTITY_FILE = DATA_DIR / 'train_identity.csv'
TEST_TRANSACTION_FILE = DATA_DIR / 'test_transaction.csv'
TEST_IDENTITY_FILE = DATA_DIR / 'test_identity.csv'

# Output files
MERGED_DATA_FILE = PROCESSED_DIR / 'train_merged.csv'
CLEANED_DATA_FILE = PROCESSED_DIR / 'train_cleaned.csv'
FINAL_DATA_FILE = PROCESSED_DIR / 'train_final.csv'

# Settings
EXTREME_MISSING_THRESHOLD = 90
RANDOM_SEED = 42

# Critical features (never drop)
CRITICAL_FEATURES = [
    'TransactionID', 'isFraud', 'TransactionDT', 'TransactionAmt',
    'ProductCD', 'has_identity'
]