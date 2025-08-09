"""
Model training pipeline for the loan approval prediction system.
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from typing import Dict, Any, Tuple, List, Optional
import joblib
from pathlib import Path
from loguru import logger
from src.config_loader import config


class ModelTrainer:
    """Handles model training, hyperparameter tuning, and model selection."""
    
    def __init__(self):
        """Initialize model trainer with configuration."""
        self.model_config = config.get('models')
        self.evaluation_config = config.get_evaluation_config()
        self.data_config = config.get_data_config()
        
        self.models = {}
        self.best_models = {}
        self.training_results = {}
        
    def get_base_models(self) -> Dict[str, Any]:
        """
        Get base model instances with default parameters.
        
        Returns:
            Dictionary of model instances
        """
        models = {
            'logistic_regression': LogisticRegression(
                random_state=self.data_config['random_state'],
                max_iter=1000
            ),
            'decision_tree': DecisionTreeClassifier(
                random_state=self.data_config['random_state']
            ),
            'random_forest': RandomForestClassifier(
                random_state=self.data_config['random_state'],
                n_jobs=-1
            ),
            'gradient_boosting': GradientBoostingClassifier(
                random_state=self.data_config['random_state']
            )
        }
        
        return models
    
    def perform_hyperparameter_tuning(self, model_name: str, model: Any, X_train: np.ndarray, y_train: np.ndarray) -> Any:
        """
        Perform hyperparameter tuning using GridSearchCV.
        
        Args:
            model_name: Name of the model
            model: Model instance
            X_train: Training features
            y_train: Training target
            
        Returns:
            Best model after hyperparameter tuning
        """
        if model_name not in self.model_config:
            logger.warning(f"No hyperparameter configuration found for {model_name}. Using default parameters.")
            return model
        
        param_grid = self.model_config[model_name]['hyperparameters']
        
        # Set up cross-validation
        cv = StratifiedKFold(
            n_splits=self.evaluation_config['cv_folds'],
            shuffle=True,
            random_state=self.data_config['random_state']
        )
        
        # Perform grid search
        grid_search = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            cv=cv,
            scoring=self.evaluation_config['primary_metric'],
            n_jobs=-1,
            verbose=1
        )
        
        logger.info(f"Starting hyperparameter tuning for {model_name}...")
        grid_search.fit(X_train, y_train)
        
        logger.info(f"Best parameters for {model_name}: {grid_search.best_params_}")
        logger.info(f"Best CV score for {model_name}: {grid_search.best_score_:.4f}")
        
        return grid_search.best_estimator_
    
    def train_single_model(self, model_name: str, X_train: np.ndarray, y_train: np.ndarray, 
                          X_val: np.ndarray, y_val: np.ndarray, tune_hyperparameters: bool = True) -> Dict[str, Any]:
        """
        Train a single model with optional hyperparameter tuning.
        
        Args:
            model_name: Name of the model to train
            X_train, y_train: Training data
            X_val, y_val: Validation data
            tune_hyperparameters: Whether to perform hyperparameter tuning
            
        Returns:
            Dictionary with training results
        """
        logger.info(f"Training {model_name}...")
        
        # Get base model
        base_models = self.get_base_models()
        if model_name not in base_models:
            raise ValueError(f"Unknown model: {model_name}")
        
        model = base_models[model_name]
        
        # Hyperparameter tuning
        if tune_hyperparameters:
            model = self.perform_hyperparameter_tuning(model_name, model, X_train, y_train)
        
        # Train the model
        model.fit(X_train, y_train)
        
        # Cross-validation scores
        cv_scores = self._get_cv_scores(model, X_train, y_train)
        
        # Validation predictions
        y_val_pred = model.predict(X_val)
        y_val_pred_proba = model.predict_proba(X_val)[:, 1] if hasattr(model, 'predict_proba') else None
        
        # Store model and results
        self.models[model_name] = model
        
        training_result = {
            'model': model,
            'cv_scores': cv_scores,
            'validation_predictions': y_val_pred,
            'validation_probabilities': y_val_pred_proba,
            'hyperparameters': model.get_params()
        }
        
        self.training_results[model_name] = training_result
        
        logger.info(f"{model_name} training completed")
        return training_result
    
    def _get_cv_scores(self, model: Any, X: np.ndarray, y: np.ndarray) -> Dict[str, List[float]]:
        """Get cross-validation scores for multiple metrics."""
        cv = StratifiedKFold(
            n_splits=self.evaluation_config['cv_folds'],
            shuffle=True,
            random_state=self.data_config['random_state']
        )
        
        cv_scores = {}
        
        for metric in self.evaluation_config['scoring_metrics']:
            scores = cross_val_score(model, X, y, cv=cv, scoring=metric)
            cv_scores[metric] = scores.tolist()
            
            logger.info(f"  {metric.upper()} CV: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")
        
        return cv_scores
    
    def train_all_models(self, X_train: np.ndarray, y_train: np.ndarray, 
                        X_val: np.ndarray, y_val: np.ndarray, 
                        models_to_train: Optional[List[str]] = None) -> Dict[str, Dict[str, Any]]:
        """
        Train all specified models.
        
        Args:
            X_train, y_train: Training data
            X_val, y_val: Validation data
            models_to_train: List of model names to train (if None, train all)
            
        Returns:
            Dictionary with all training results
        """
        if models_to_train is None:
            models_to_train = list(self.get_base_models().keys())
        
        logger.info(f"Training {len(models_to_train)} models: {models_to_train}")
        
        all_results = {}
        
        for model_name in models_to_train:
            try:
                result = self.train_single_model(model_name, X_train, y_train, X_val, y_val)
                all_results[model_name] = result
            except Exception as e:
                logger.error(f"Failed to train {model_name}: {e}")
                continue
        
        logger.info("All model training completed")
        return all_results
    
    def select_best_model(self, validation_results: Dict[str, Any]) -> Tuple[str, Any]:
        """
        Select the best model based on validation performance.
        
        Args:
            validation_results: Results from model evaluation
            
        Returns:
            Tuple of (best_model_name, best_model_instance)
        """
        primary_metric = self.evaluation_config['primary_metric']
        
        best_score = -np.inf
        best_model_name = None
        
        for model_name, results in validation_results.items():
            if primary_metric in results:
                score = results[primary_metric]
                if score > best_score:
                    best_score = score
                    best_model_name = model_name
        
        if best_model_name is None:
            raise ValueError("No valid model found for selection")
        
        best_model = self.models[best_model_name]
        
        logger.info(f"Best model selected: {best_model_name} with {primary_metric}: {best_score:.4f}")
        
        return best_model_name, best_model
    
    def save_model(self, model: Any, model_name: str, metadata: Dict[str, Any]) -> str:
        """
        Save trained model with metadata for production use.
        
        Args:
            model: Trained model instance
            model_name: Name of the model
            metadata: Additional metadata to save
            
        Returns:
            Path where model was saved
        """
        models_dir = Path("models")
        models_dir.mkdir(exist_ok=True)
        
        # Create model artifact
        model_artifact = {
            'model': model,
            'model_name': model_name,
            'metadata': metadata,
            'feature_names': getattr(self, 'feature_names', None),
            'model_version': config.get('production.model_version', 'v1.0')
        }
        
        # Save model
        model_path = models_dir / f"{model_name}_model.joblib"
        joblib.dump(model_artifact, model_path)
        
        logger.info(f"Model saved to {model_path}")
        return str(model_path)
    
    def load_model(self, model_path: str) -> Tuple[Any, Dict[str, Any]]:
        """
        Load a saved model.
        
        Args:
            model_path: Path to the saved model
            
        Returns:
            Tuple of (model_instance, metadata)
        """
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        model_artifact = joblib.load(model_path)
        
        logger.info(f"Model loaded from {model_path}")
        return model_artifact['model'], model_artifact['metadata']
