from src.preprocessing.preprocessing_pipeline import run_full_pipeline

if __name__ == "__main__":
    X_train, y_train, X_val, y_val = run_full_pipeline()
    print(f"X_train: {X_train.shape} | X_val: {X_val.shape}")