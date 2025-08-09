"""
Data loading utilities for the sales forecasting project.
"""

import pandas as pd
from pathlib import Path
from typing import Tuple, Optional
import sys
sys.path.append(str(Path(__file__).parent.parent))

from utils.config import (
    TRAIN_FILE, TEST_FILE, FEATURES_FILE, STORES_FILE,
    PROCESSED_TRAIN_FILE
)
from utils.logger import get_project_logger

logger = get_project_logger("data_loader")

class DataLoader:
    """Data loading and basic validation utilities."""
    
    def __init__(self):
        self.logger = logger
    
    def load_raw_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Load all raw datasets.
        
        Returns:
            Tuple of (train_df, test_df, features_df, stores_df)
        """
        self.logger.info("Loading raw datasets...")
        
        try:
            train_df = pd.read_csv(TRAIN_FILE)
            test_df = pd.read_csv(TEST_FILE)
            features_df = pd.read_csv(FEATURES_FILE)
            stores_df = pd.read_csv(STORES_FILE)
            
            # Convert date columns
            for df in [train_df, test_df, features_df]:
                if 'Date' in df.columns:
                    df['Date'] = pd.to_datetime(df['Date'])
            
            self.logger.info(f"Raw data loaded successfully:")
            self.logger.info(f"  Train: {train_df.shape}")
            self.logger.info(f"  Test: {test_df.shape}")
            self.logger.info(f"  Features: {features_df.shape}")
            self.logger.info(f"  Stores: {stores_df.shape}")
            
            return train_df, test_df, features_df, stores_df
            
        except Exception as e:
            self.logger.error(f"Error loading raw data: {str(e)}")
            raise
    
    def load_processed_data(self) -> Optional[pd.DataFrame]:
        """
        Load processed training data if available.
        
        Returns:
            Processed DataFrame or None if not found
        """
        if PROCESSED_TRAIN_FILE.exists():
            self.logger.info("Loading processed training data...")
            try:
                df = pd.read_csv(PROCESSED_TRAIN_FILE)
                df['Date'] = pd.to_datetime(df['Date'])
                self.logger.info(f"Processed data loaded: {df.shape}")
                return df
            except Exception as e:
                self.logger.error(f"Error loading processed data: {str(e)}")
                return None
        else:
            self.logger.warning("Processed data file not found")
            return None
    
    def validate_data(self, df: pd.DataFrame, dataset_name: str) -> bool:
        """
        Perform basic data validation.
        
        Args:
            df: DataFrame to validate
            dataset_name: Name of the dataset for logging
        
        Returns:
            True if validation passes, False otherwise
        """
        self.logger.info(f"Validating {dataset_name} dataset...")
        
        # Check for empty dataset
        if df.empty:
            self.logger.error(f"{dataset_name} dataset is empty")
            return False
        
        # Check for required columns
        required_cols = {
            'train': ['Store', 'Dept', 'Date', 'Weekly_Sales'],
            'test': ['Store', 'Dept', 'Date'],
            'features': ['Store', 'Date'],
            'stores': ['Store', 'Type', 'Size']
        }
        
        if dataset_name.lower() in required_cols:
            missing_cols = set(required_cols[dataset_name.lower()]) - set(df.columns)
            if missing_cols:
                self.logger.error(f"{dataset_name} missing required columns: {missing_cols}")
                return False
        
        # Check data types
        if 'Date' in df.columns and not pd.api.types.is_datetime64_any_dtype(df['Date']):
            self.logger.warning(f"{dataset_name} Date column is not datetime type")
        
        self.logger.info(f"{dataset_name} validation passed")
        return True

# Example usage
if __name__ == "__main__":
    loader = DataLoader()
    train_df, test_df, features_df, stores_df = loader.load_raw_data()
    
    # Validate datasets
    loader.validate_data(train_df, "train")
    loader.validate_data(test_df, "test")
    loader.validate_data(features_df, "features")
    loader.validate_data(stores_df, "stores")
