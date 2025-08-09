"""
Data preprocessing pipeline for the loan approval prediction system.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from typing import Tuple, Dict, Any, Optional
from pathlib import Path
import joblib
from loguru import logger
from src.config_loader import config


class LoanDataPreprocessor:
    """Comprehensive data preprocessing pipeline for loan approval prediction."""
    
    def __init__(self):
        """Initialize preprocessor with configuration."""
        self.feature_config = config.get('features')
        self.preprocessing_config = config.get_preprocessing_config()
        self.data_config = config.get_data_config()
        
        self.categorical_features = self.feature_config['categorical_features']
        self.numerical_features = self.feature_config['numerical_features']
        self.target_col = self.feature_config['target_column']
        self.id_col = self.feature_config['id_column']
        
        self.preprocessor = None
        self.label_encoder = None
        self.feature_names = None
        
    def create_preprocessing_pipeline(self) -> ColumnTransformer:
        """
        Create sklearn preprocessing pipeline.
        
        Returns:
            Configured ColumnTransformer pipeline
        """
        # Numerical preprocessing pipeline
        numerical_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy=self.preprocessing_config['missing_value_strategy']['numerical'])),
            ('scaler', self._get_scaler())
        ])
        
        # Categorical preprocessing pipeline
        categorical_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy=self.preprocessing_config['missing_value_strategy']['categorical'])),
            ('encoder', self._get_encoder())
        ])
        
        # Combine preprocessing steps
        preprocessor = ColumnTransformer([
            ('numerical', numerical_pipeline, self.numerical_features),
            ('categorical', categorical_pipeline, self.categorical_features)
        ])
        
        return preprocessor
    
    def _get_scaler(self):
        """Get the appropriate scaler based on configuration."""
        scaler_type = self.preprocessing_config['scaling_method']
        
        scalers = {
            'standard': StandardScaler(),
            'minmax': MinMaxScaler(),
            'robust': RobustScaler()
        }
        
        return scalers.get(scaler_type, StandardScaler())
    
    def _get_encoder(self):
        """Get the appropriate encoder based on configuration."""
        encoding_method = self.preprocessing_config['encoding_method']
        
        if encoding_method == 'onehot':
            return OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')
        else:
            return LabelEncoder()
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create engineered features for better model performance.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with engineered features
        """
        df_engineered = df.copy()
        
        # Financial ratios
        df_engineered['debt_to_income_ratio'] = df_engineered['loan_amount'] / df_engineered['income_annum']
        df_engineered['loan_to_asset_ratio'] = df_engineered['loan_amount'] / (
            df_engineered['residential_assets_value'] + 
            df_engineered['commercial_assets_value'] + 
            df_engineered['luxury_assets_value'] + 
            df_engineered['bank_asset_value'] + 1  # Add 1 to avoid division by zero
        )
        
        # Total assets
        df_engineered['total_assets'] = (
            df_engineered['residential_assets_value'] + 
            df_engineered['commercial_assets_value'] + 
            df_engineered['luxury_assets_value'] + 
            df_engineered['bank_asset_value']
        )
        
        # Asset diversity (number of non-zero asset types)
        asset_columns = ['residential_assets_value', 'commercial_assets_value', 
                        'luxury_assets_value', 'bank_asset_value']
        df_engineered['asset_diversity'] = (df_engineered[asset_columns] > 0).sum(axis=1)
        
        # Credit score categories
        df_engineered['credit_score_category'] = pd.cut(
            df_engineered['cibil_score'], 
            bins=[0, 550, 650, 750, 900], 
            labels=['Poor', 'Fair', 'Good', 'Excellent']
        )
        
        # Income categories
        df_engineered['income_category'] = pd.cut(
            df_engineered['income_annum'], 
            bins=[0, 2000000, 5000000, 10000000, float('inf')], 
            labels=['Low', 'Medium', 'High', 'Very High']
        )
        
        # Update feature lists to include engineered features
        new_numerical_features = ['debt_to_income_ratio', 'loan_to_asset_ratio', 'total_assets', 'asset_diversity']
        new_categorical_features = ['credit_score_category', 'income_category']
        
        self.numerical_features.extend(new_numerical_features)
        self.categorical_features.extend(new_categorical_features)
        
        logger.info(f"Feature engineering completed. Added {len(new_numerical_features + new_categorical_features)} new features")
        
        return df_engineered
    
    def prepare_target_variable(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare target variable for modeling.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with encoded target variable
        """
        df_processed = df.copy()
        
        # Encode target variable (Approved=1, Rejected=0)
        self.label_encoder = LabelEncoder()
        df_processed[self.target_col + '_encoded'] = self.label_encoder.fit_transform(df_processed[self.target_col])
        
        logger.info(f"Target variable encoded: {dict(zip(self.label_encoder.classes_, self.label_encoder.transform(self.label_encoder.classes_)))}")
        
        return df_processed
    
    def split_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Split data into train, validation, and test sets.
        
        Args:
            df: Preprocessed DataFrame
            
        Returns:
            X_train, X_val, X_test, y_train, y_val, y_test
        """
        # Prepare features and target
        feature_columns = self.numerical_features + self.categorical_features
        X = df[feature_columns]
        y = df[self.target_col + '_encoded']
        
        # First split: train+val vs test
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, 
            test_size=self.data_config['test_size'],
            random_state=self.data_config['random_state'],
            stratify=y
        )
        
        # Second split: train vs validation
        val_size = self.data_config['validation_size'] / (1 - self.data_config['test_size'])
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp,
            test_size=val_size,
            random_state=self.data_config['random_state'],
            stratify=y_temp
        )
        
        logger.info(f"Data split completed:")
        logger.info(f"  Train: {X_train.shape[0]} samples")
        logger.info(f"  Validation: {X_val.shape[0]} samples") 
        logger.info(f"  Test: {X_test.shape[0]} samples")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def fit_transform_pipeline(self, X_train: pd.DataFrame, X_val: pd.DataFrame, X_test: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Fit preprocessing pipeline on training data and transform all sets.
        
        Args:
            X_train, X_val, X_test: Feature DataFrames
            
        Returns:
            Transformed feature arrays
        """
        # Create and fit preprocessing pipeline
        self.preprocessor = self.create_preprocessing_pipeline()
        
        # Fit on training data and transform all sets
        X_train_processed = self.preprocessor.fit_transform(X_train)
        X_val_processed = self.preprocessor.transform(X_val)
        X_test_processed = self.preprocessor.transform(X_test)
        
        # Store feature names for later use
        self._store_feature_names()
        
        logger.info("Preprocessing pipeline fitted and applied successfully")
        logger.info(f"Processed feature shape: {X_train_processed.shape}")
        
        return X_train_processed, X_val_processed, X_test_processed
    
    def _store_feature_names(self) -> None:
        """Store feature names after preprocessing for interpretability."""
        feature_names = []
        
        # Numerical feature names (unchanged)
        feature_names.extend(self.numerical_features)
        
        # Categorical feature names (depends on encoding method)
        if self.preprocessing_config['encoding_method'] == 'onehot':
            # Get feature names from OneHotEncoder
            categorical_transformer = self.preprocessor.named_transformers_['categorical']
            encoder = categorical_transformer.named_steps['encoder']
            
            if hasattr(encoder, 'get_feature_names_out'):
                categorical_names = encoder.get_feature_names_out(self.categorical_features)
                feature_names.extend(categorical_names)
            else:
                # Fallback for older sklearn versions
                feature_names.extend(self.categorical_features)
        else:
            feature_names.extend(self.categorical_features)
        
        self.feature_names = feature_names
        logger.info(f"Stored {len(self.feature_names)} feature names")
    
    def save_preprocessor(self, filepath: str = "models/preprocessor.joblib") -> None:
        """Save the fitted preprocessor for production use."""
        if self.preprocessor is None:
            raise ValueError("Preprocessor must be fitted before saving")
        
        save_path = Path(filepath)
        save_path.parent.mkdir(exist_ok=True)
        
        # Save preprocessor and related objects
        preprocessing_artifacts = {
            'preprocessor': self.preprocessor,
            'label_encoder': self.label_encoder,
            'feature_names': self.feature_names,
            'numerical_features': self.numerical_features,
            'categorical_features': self.categorical_features
        }
        
        joblib.dump(preprocessing_artifacts, save_path)
        logger.info(f"Preprocessor saved to {save_path}")
    
    def load_preprocessor(self, filepath: str = "models/preprocessor.joblib") -> None:
        """Load a previously saved preprocessor."""
        load_path = Path(filepath)
        
        if not load_path.exists():
            raise FileNotFoundError(f"Preprocessor file not found: {load_path}")
        
        artifacts = joblib.load(load_path)
        
        self.preprocessor = artifacts['preprocessor']
        self.label_encoder = artifacts['label_encoder']
        self.feature_names = artifacts['feature_names']
        self.numerical_features = artifacts['numerical_features']
        self.categorical_features = artifacts['categorical_features']
        
        logger.info(f"Preprocessor loaded from {load_path}")
    
    def process_full_pipeline(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Run the complete preprocessing pipeline.
        
        Args:
            df: Raw DataFrame
            
        Returns:
            X_train, X_val, X_test, y_train, y_val, y_test (all processed)
        """
        logger.info("Starting complete preprocessing pipeline...")
        
        # Feature engineering
        df_engineered = self.engineer_features(df)
        
        # Prepare target variable
        df_processed = self.prepare_target_variable(df_engineered)
        
        # Split data
        X_train, X_val, X_test, y_train, y_val, y_test = self.split_data(df_processed)
        
        # Fit and transform preprocessing pipeline
        X_train_processed, X_val_processed, X_test_processed = self.fit_transform_pipeline(X_train, X_val, X_test)
        
        # Save preprocessor for production
        self.save_preprocessor()
        
        logger.info("Complete preprocessing pipeline finished successfully")
        
        return X_train_processed, X_val_processed, X_test_processed, y_train, y_val, y_test
