"""
Data loading and initial validation for the loan approval prediction system.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, Any
from loguru import logger
from src.config_loader import config


class DataLoader:
    """Handles data loading and initial validation."""
    
    def __init__(self):
        """Initialize the data loader with configuration."""
        self.data_config = config.get_data_config()
        self.feature_config = config.get('features')
        
    def load_raw_data(self) -> pd.DataFrame:
        """
        Load raw data from CSV file.

        Returns:
            Raw DataFrame
        """
        try:
            data_path = Path(self.data_config['raw_data_path'])
            df = pd.read_csv(data_path)

            # Clean column names (remove extra spaces)
            df.columns = df.columns.str.strip()

            # Clean string data (remove leading/trailing whitespace)
            for col in df.select_dtypes(include=['object']).columns:
                df[col] = df[col].astype(str).str.strip()

            # Additional cleaning for target column specifically
            target_col = self.feature_config['target_column']
            if target_col in df.columns:
                df[target_col] = df[target_col].str.strip()
                logger.info(f"Target values after cleaning: {df[target_col].unique()}")

            logger.info(f"Cleaned column names: {list(df.columns)}")

            logger.info(f"Data loaded successfully. Shape: {df.shape}")
            logger.info(f"Columns: {list(df.columns)}")

            return df
            
        except FileNotFoundError:
            logger.error(f"Data file not found: {data_path}")
            raise
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise
    
    def validate_data_schema(self, df: pd.DataFrame) -> bool:
        """
        Validate that the loaded data has expected columns and structure.
        
        Args:
            df: DataFrame to validate
            
        Returns:
            True if validation passes
        """
        expected_columns = (
            self.feature_config['categorical_features'] +
            self.feature_config['numerical_features'] +
            [self.feature_config['target_column'], self.feature_config['id_column']]
        )
        
        missing_columns = set(expected_columns) - set(df.columns)
        extra_columns = set(df.columns) - set(expected_columns)
        
        if missing_columns:
            logger.error(f"Missing expected columns: {missing_columns}")
            return False
            
        if extra_columns:
            logger.warning(f"Extra columns found: {extra_columns}")
        
        # Validate target column values
        target_col = self.feature_config['target_column']
        unique_targets = df[target_col].unique()
        expected_targets = ['Approved', 'Rejected']
        
        if not all(target in expected_targets for target in unique_targets):
            logger.error(f"Unexpected target values: {unique_targets}")
            return False
        
        logger.info("Data schema validation passed")
        return True
    
    def get_data_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate comprehensive data summary.
        
        Args:
            df: DataFrame to summarize
            
        Returns:
            Dictionary with data summary statistics
        """
        summary = {
            'shape': df.shape,
            'columns': list(df.columns),
            'dtypes': df.dtypes.to_dict(),
            'missing_values': df.isnull().sum().to_dict(),
            'missing_percentage': (df.isnull().sum() / len(df) * 100).to_dict(),
            'target_distribution': df[self.feature_config['target_column']].value_counts().to_dict(),
            'numerical_summary': df.describe().to_dict(),
            'categorical_summary': {}
        }
        
        # Add categorical feature summaries
        for col in self.feature_config['categorical_features']:
            if col in df.columns:
                summary['categorical_summary'][col] = df[col].value_counts().to_dict()
        
        return summary
    
    def load_and_validate(self) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Load data and perform initial validation.
        
        Returns:
            Tuple of (DataFrame, summary_dict)
        """
        df = self.load_raw_data()
        
        if not self.validate_data_schema(df):
            raise ValueError("Data schema validation failed")
        
        summary = self.get_data_summary(df)
        
        logger.info("Data loading and validation completed successfully")
        return df, summary
