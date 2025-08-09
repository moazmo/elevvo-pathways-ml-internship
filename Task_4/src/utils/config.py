"""
Configuration management for the sales forecasting project.
"""

import os
from pathlib import Path
from typing import Dict, Any

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"
RESULTS_DIR = PROJECT_ROOT / "results"
MODELS_DIR = RESULTS_DIR / "models"
PLOTS_DIR = RESULTS_DIR / "plots"
REPORTS_DIR = RESULTS_DIR / "reports"
CONFIGS_DIR = PROJECT_ROOT / "configs"

# Data files
TRAIN_FILE = RAW_DATA_DIR / "train.csv"
TEST_FILE = RAW_DATA_DIR / "test.csv"
FEATURES_FILE = RAW_DATA_DIR / "features.csv"
STORES_FILE = RAW_DATA_DIR / "stores.csv"

# Processed data files
PROCESSED_TRAIN_FILE = PROCESSED_DATA_DIR / "train_processed.csv"
FEATURE_LIST_FILE = PROCESSED_DATA_DIR / "feature_list.txt"
LABEL_ENCODERS_FILE = PROCESSED_DATA_DIR / "label_encoders.pkl"

# Model configuration
MODEL_CONFIG = {
    "random_state": 42,
    "test_size": 0.2,
    "cv_folds": 5,
    "lag_periods": [1, 2, 4, 8, 12, 26],
    "rolling_windows": [4, 8, 12, 26],
    "target_column": "Weekly_Sales"
}

# Feature engineering configuration
FEATURE_CONFIG = {
    "handle_negative_sales": True,
    "outlier_threshold": 3,  # standard deviations
    "min_periods_rolling": 1,
    "ewm_spans": [4, 8, 12, 26]
}

# Evaluation metrics
METRICS_CONFIG = {
    "primary_metric": "wmae",  # Weighted Mean Absolute Error
    "secondary_metrics": ["mae", "rmse", "mape"],
    "holiday_weight": 5,  # Weight for holiday weeks in WMAE
    "regular_weight": 1   # Weight for regular weeks in WMAE
}

def ensure_directories():
    """Create necessary directories if they don't exist."""
    directories = [
        DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, EXTERNAL_DATA_DIR,
        RESULTS_DIR, MODELS_DIR, PLOTS_DIR, REPORTS_DIR, CONFIGS_DIR
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
    
    print("All necessary directories created/verified.")

if __name__ == "__main__":
    ensure_directories()
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Data directory: {DATA_DIR}")
    print(f"Results directory: {RESULTS_DIR}")
