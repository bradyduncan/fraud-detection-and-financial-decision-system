# fraud-detection-and-financial-decision-system
Intelligent Fraud Detection and Financial Decision Systems Using Ensemble Learning

An end-to-end machine learning pipeline for detecting fraudulent transactions on the
[IEEE-CIS Fraud Detection dataset](https://www.kaggle.com/c/ieee-fraud-detection).
Built as a DS 5500 Capstone project by Brady Duncan and Vedashree Bane.

---

## Project Overview

Financial fraud costs institutions billions annually. This project builds a production-grade
fraud detection system that flags suspicious transactions in real time. The pipeline covers
data ingestion, temporal feature engineering, ensemble model training, and an interactive
Streamlit dashboard for exploring predictions and model explainability.

**Dataset:** IEEE-CIS Fraud Detection (Kaggle) — 590,000 transactions, 430 features, ~3.5% fraud rate.

---

## Why Gradient Boosting (not Deep Learning or LLMs)?

This is a structured tabular dataset, which is the canonical domain for gradient boosting methods:

| Approach | Verdict | Reason |
|----------|---------|--------|
| **Gradient Boosting (chosen)** | Best fit | Tabular data, ~590k rows, handles mixed types, outlier-robust, interpretable feature importances |
| Deep Learning (MLP/RNN) | Not suitable | No sequential structure in individual transactions; insufficient data for DL to outperform boosting on tabular tasks |
| LLMs | Not applicable | Designed for natural language; our features are numeric/categorical, not text |

Four gradient boosting variants were evaluated — LightGBM, XGBoost, CatBoost, and
HistGradientBoosting — to identify the best-performing model on the imbalanced fraud class.

---

## Project Structure

```
fraud-detection-and-financial-decision-system/
├── main.py                          # Entry point: runs full preprocessing pipeline
├── app.py                           # Streamlit dashboard entry point
├── requirements.txt
├── src/
│   ├── config.py                    # Paths and global constants
│   ├── preprocessing/               # Data loading, cleaning, feature engineering
│   │   ├── data_loader_merger.py
│   │   ├── missing_handler.py
│   │   ├── encoder.py
│   │   ├── feature_selection.py
│   │   ├── temporal_features.py
│   │   └── preprocessing_pipeline.py
│   ├── fraud_detection_models/      # Model training and evaluation
│   │   ├── lightgbm_model.py
│   │   ├── xgboost_model.py
│   │   ├── catboost_model.py
│   │   ├── histgb_model.py
│   │   └── experimentation.py
│   └── utils/
│       ├── data_splits.py
│       ├── normalisation.py
│       └── smote.py
├── data/
│   ├── train_transaction.csv
│   ├── train_identity.csv
│   └── processed/                   # Generated artifacts (splits, models)
├── presentations/                   # Demo notebooks
└── tests/                           # Pytest unit and integration tests
```

---

## Setup Instructions

### Windows
```powershell
.\init.ps1
```

### Mac / Linux
```bash
make init
# If 'make' is not found on Mac: xcode-select --install
```

### Manual (any platform)
```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

---

## Execution

### 1. Run the preprocessing pipeline
```bash
python main.py
```
Outputs preprocessed splits to `data/processed/splits/`.

### 2. Train a model
```bash
python -m src.fraud_detection_models.lightgbm_model
python -m src.fraud_detection_models.xgboost_model
python -m src.fraud_detection_models.catboost_model
python -m src.fraud_detection_models.histgb_model
```

### 3. Launch the Streamlit dashboard
```bash
streamlit run app.py
```

### 4. Run tests
```bash
pytest tests/ -v
```

---

## Reproducibility

- Random seed fixed at `42` in `src/config.py` and in all model configurations.
- All file paths are relative via `pathlib.Path(__file__).parent` — no hard-coded paths.
- Preprocessing artifacts (imputer, encoder, feature list) are saved to `data/processed/models/`
  and reused by model training scripts, ensuring identical splits across all four models.
