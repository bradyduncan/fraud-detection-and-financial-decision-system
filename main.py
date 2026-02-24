from src.preprocessing.preprocessing_pipeline import run_full_pipeline

if __name__ == "__main__":
    df = run_full_pipeline(save_intermediate=True)
    print(f"Final shape: {df.shape}")