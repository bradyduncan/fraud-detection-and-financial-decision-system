from src.preprocessing.preprocessing_pipeline import run_full_pipeline
import pandas as pd

if __name__ == "__main__":
    # Run pipeline and save to file
    df = run_full_pipeline(save_intermediate=False)
    df.to_csv("/data/processed_data.csv", index=False)
    
    # make sure df exists
    print(df.shape)