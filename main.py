from src.preprocessing.preprocessing_pipeline import run_full_pipeline

if __name__ == "__main__":
    # Run pipeline
    df = run_full_pipeline(save_intermediate=False)
    
    # make sure df exists
    print(df.shape)