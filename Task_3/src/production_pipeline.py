"""
Production-ready inference pipeline for loan approval prediction.
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
from pydantic import BaseModel, Field, validator
from loguru import logger
import json
from datetime import datetime
from src.config_loader import config


class LoanApplicationRequest(BaseModel):
    """Pydantic model for loan application request validation."""
    
    loan_id: Optional[str] = Field(None, description="Unique loan application ID")
    no_of_dependents: int = Field(..., ge=0, le=10, description="Number of dependents")
    education: str = Field(..., description="Education level (Graduate/Not Graduate)")
    self_employed: str = Field(..., description="Self employment status (Yes/No)")
    income_annum: float = Field(..., gt=0, description="Annual income")
    loan_amount: float = Field(..., gt=0, description="Requested loan amount")
    loan_term: int = Field(..., gt=0, le=30, description="Loan term in years")
    cibil_score: int = Field(..., ge=300, le=900, description="CIBIL credit score")
    residential_assets_value: float = Field(..., ge=0, description="Residential assets value")
    commercial_assets_value: float = Field(..., ge=0, description="Commercial assets value")
    luxury_assets_value: float = Field(..., ge=0, description="Luxury assets value")
    bank_asset_value: float = Field(..., ge=0, description="Bank assets value")
    
    @validator('education')
    def validate_education(cls, v):
        if v not in ['Graduate', 'Not Graduate']:
            raise ValueError('Education must be "Graduate" or "Not Graduate"')
        return v
    
    @validator('self_employed')
    def validate_self_employed(cls, v):
        if v not in ['Yes', 'No']:
            raise ValueError('Self employed must be "Yes" or "No"')
        return v


class LoanPredictionResponse(BaseModel):
    """Pydantic model for loan prediction response."""
    
    loan_id: Optional[str]
    prediction: str = Field(..., description="Approved or Rejected")
    confidence: float = Field(..., ge=0, le=1, description="Prediction confidence")
    risk_score: float = Field(..., ge=0, le=1, description="Risk score (0=low risk, 1=high risk)")
    key_factors: List[str] = Field(..., description="Key factors influencing the decision")
    timestamp: str = Field(..., description="Prediction timestamp")
    model_version: str = Field(..., description="Model version used")


class ProductionPipeline:
    """Production-ready inference pipeline for loan approval prediction."""
    
    def __init__(self, model_path: Optional[str] = None, preprocessor_path: Optional[str] = None):
        """
        Initialize production pipeline.
        
        Args:
            model_path: Path to saved model
            preprocessor_path: Path to saved preprocessor
        """
        self.model_path = model_path or "models/decision_tree_model.joblib"
        self.preprocessor_path = preprocessor_path or "models/preprocessor.joblib"
        
        self.model = None
        self.preprocessor = None
        self.feature_names = None
        self.label_encoder = None
        self.model_metadata = None
        self.numerical_features = None
        self.categorical_features = None
        
        self._load_artifacts()
    
    def _load_artifacts(self) -> None:
        """Load model and preprocessing artifacts."""
        try:
            # Load preprocessor
            if Path(self.preprocessor_path).exists():
                preprocessor_artifacts = joblib.load(self.preprocessor_path)
                self.preprocessor = preprocessor_artifacts['preprocessor']
                self.label_encoder = preprocessor_artifacts['label_encoder']
                self.feature_names = preprocessor_artifacts['feature_names']
                self.numerical_features = preprocessor_artifacts['numerical_features']
                self.categorical_features = preprocessor_artifacts['categorical_features']
                logger.info("Preprocessor loaded successfully")
            else:
                logger.warning(f"Preprocessor not found at {self.preprocessor_path}")
            
            # Load model
            if Path(self.model_path).exists():
                model_artifacts = joblib.load(self.model_path)
                self.model = model_artifacts['model']
                self.model_metadata = model_artifacts.get('metadata', {})
                logger.info("Model loaded successfully")
            else:
                logger.warning(f"Model not found at {self.model_path}")
                
        except Exception as e:
            logger.error(f"Error loading artifacts: {e}")
            raise
    
    def _validate_input(self, application_data: Union[Dict[str, Any], pd.DataFrame, pd.Series]) -> LoanApplicationRequest:
        """
        Validate input data using Pydantic model.

        Args:
            application_data: Raw application data (dict, DataFrame, or Series)

        Returns:
            Validated LoanApplicationRequest
        """
        try:
            # Convert pandas DataFrame/Series to dictionary
            if isinstance(application_data, pd.DataFrame):
                if len(application_data) != 1:
                    raise ValueError("DataFrame must contain exactly one row for single prediction")
                application_data = application_data.iloc[0].to_dict()
            elif isinstance(application_data, pd.Series):
                application_data = application_data.to_dict()

            # Ensure all values are Python native types (not numpy/pandas types)
            clean_data = {}
            for key, value in application_data.items():
                if pd.isna(value):
                    clean_data[key] = None
                elif hasattr(value, 'item'):  # numpy scalar
                    clean_data[key] = value.item()
                else:
                    clean_data[key] = value

            return LoanApplicationRequest(**clean_data)
        except Exception as e:
            logger.error(f"Input validation failed: {e}")
            raise ValueError(f"Invalid input data: {e}")
    
    def _preprocess_input(self, validated_data: LoanApplicationRequest) -> np.ndarray:
        """
        Preprocess input data for model inference.

        Args:
            validated_data: Validated loan application data

        Returns:
            Preprocessed feature array
        """
        if self.preprocessor is None:
            raise ValueError("Preprocessor not loaded")

        # Convert to DataFrame
        data_dict = validated_data.dict()
        df = pd.DataFrame([data_dict])

        # Apply feature engineering (same as training)
        df_engineered = self._engineer_features_inference(df)

        # Use the exact feature names and order from training
        if hasattr(self, 'numerical_features') and hasattr(self, 'categorical_features'):
            # Use the features from the loaded preprocessor (exact same as training)
            all_features = self.numerical_features + self.categorical_features
        else:
            logger.error("Feature names not loaded from preprocessor")
            raise ValueError("Preprocessor feature names not available")

        # Ensure all required features are present
        missing_features = set(all_features) - set(df_engineered.columns)
        if missing_features:
            logger.error(f"Missing features for inference: {missing_features}")
            logger.error(f"Available features: {list(df_engineered.columns)}")
            logger.error(f"Required features: {all_features}")
            raise ValueError(f"Missing required features: {missing_features}")

        # Select and order features exactly as in training
        X = df_engineered[all_features]

        # Apply preprocessing
        X_processed = self.preprocessor.transform(X)

        return X_processed
    
    def _engineer_features_inference(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply the same feature engineering as in training."""
        df_engineered = df.copy()

        # Financial ratios
        df_engineered['debt_to_income_ratio'] = df_engineered['loan_amount'] / df_engineered['income_annum']
        df_engineered['loan_to_asset_ratio'] = df_engineered['loan_amount'] / (
            df_engineered['residential_assets_value'] +
            df_engineered['commercial_assets_value'] +
            df_engineered['luxury_assets_value'] +
            df_engineered['bank_asset_value'] + 1
        )

        # Total assets
        df_engineered['total_assets'] = (
            df_engineered['residential_assets_value'] +
            df_engineered['commercial_assets_value'] +
            df_engineered['luxury_assets_value'] +
            df_engineered['bank_asset_value']
        )

        # Asset diversity
        asset_columns = ['residential_assets_value', 'commercial_assets_value',
                        'luxury_assets_value', 'bank_asset_value']
        df_engineered['asset_diversity'] = (df_engineered[asset_columns] > 0).sum(axis=1)

        # Credit score categories - handle edge cases
        try:
            df_engineered['credit_score_category'] = pd.cut(
                df_engineered['cibil_score'],
                bins=[0, 550, 650, 750, 900],
                labels=['Poor', 'Fair', 'Good', 'Excellent'],
                include_lowest=True
            )
            # Convert to string to avoid category issues
            df_engineered['credit_score_category'] = df_engineered['credit_score_category'].astype(str)
        except Exception as e:
            logger.warning(f"Credit score categorization failed: {e}. Using fallback.")
            # Fallback: simple categorization
            df_engineered['credit_score_category'] = 'Good'  # Default category

        # Income categories - handle edge cases
        try:
            df_engineered['income_category'] = pd.cut(
                df_engineered['income_annum'],
                bins=[0, 2000000, 5000000, 10000000, float('inf')],
                labels=['Low', 'Medium', 'High', 'Very High'],
                include_lowest=True
            )
            # Convert to string to avoid category issues
            df_engineered['income_category'] = df_engineered['income_category'].astype(str)
        except Exception as e:
            logger.warning(f"Income categorization failed: {e}. Using fallback.")
            # Fallback: simple categorization
            df_engineered['income_category'] = 'Medium'  # Default category

        return df_engineered
    
    def _get_key_factors(self, X_processed: np.ndarray, prediction_proba: float) -> List[str]:
        """
        Identify key factors influencing the prediction.
        
        Args:
            X_processed: Processed feature array
            prediction_proba: Prediction probability
            
        Returns:
            List of key factors
        """
        key_factors = []
        
        # Get feature importance or coefficients
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
            # Get top 3 most important features
            top_indices = np.argsort(importances)[-3:][::-1]
            
            if self.feature_names:
                key_factors = [self.feature_names[i] for i in top_indices]
            else:
                key_factors = [f"Feature_{i}" for i in top_indices]
                
        elif hasattr(self.model, 'coef_'):
            coefficients = self.model.coef_[0] if self.model.coef_.ndim > 1 else self.model.coef_
            # Get top 3 features by absolute coefficient value
            top_indices = np.argsort(np.abs(coefficients))[-3:][::-1]
            
            if self.feature_names:
                key_factors = [self.feature_names[i] for i in top_indices]
            else:
                key_factors = [f"Feature_{i}" for i in top_indices]
        
        # Add confidence-based factor
        if prediction_proba > 0.8:
            key_factors.append("High confidence prediction")
        elif prediction_proba < 0.6:
            key_factors.append("Low confidence prediction")
        
        return key_factors[:5]  # Return top 5 factors
    
    def predict(self, application_data: Dict[str, Any]) -> LoanPredictionResponse:
        """
        Make loan approval prediction for a single application.
        
        Args:
            application_data: Loan application data
            
        Returns:
            LoanPredictionResponse with prediction and metadata
        """
        if self.model is None:
            raise ValueError("Model not loaded")
        
        # Validate input
        validated_data = self._validate_input(application_data)
        
        # Preprocess
        X_processed = self._preprocess_input(validated_data)
        
        # Make prediction
        prediction = self.model.predict(X_processed)[0]
        prediction_proba = self.model.predict_proba(X_processed)[0] if hasattr(self.model, 'predict_proba') else [0.5, 0.5]
        
        # Convert prediction to label
        if self.label_encoder:
            prediction_label = self.label_encoder.inverse_transform([prediction])[0]
        else:
            prediction_label = "Approved" if prediction == 1 else "Rejected"
        
        # Calculate confidence and risk score
        confidence = max(prediction_proba)
        risk_score = 1 - prediction_proba[1]  # Higher risk = lower approval probability
        
        # Get key factors
        key_factors = self._get_key_factors(X_processed, prediction_proba[1])
        
        # Create response
        response = LoanPredictionResponse(
            loan_id=validated_data.loan_id,
            prediction=prediction_label,
            confidence=float(confidence),
            risk_score=float(risk_score),
            key_factors=key_factors,
            timestamp=datetime.now().isoformat(),
            model_version=config.get('production.model_version', 'v1.0')
        )
        
        logger.info(f"Prediction made: {prediction_label} (confidence: {confidence:.3f})")
        
        return response
    
    def predict_batch(self, applications: List[Dict[str, Any]]) -> List[LoanPredictionResponse]:
        """
        Make predictions for multiple loan applications.
        
        Args:
            applications: List of loan application data
            
        Returns:
            List of LoanPredictionResponse objects
        """
        logger.info(f"Processing batch of {len(applications)} applications")
        
        responses = []
        
        for i, app_data in enumerate(applications):
            try:
                response = self.predict(app_data)
                responses.append(response)
            except Exception as e:
                logger.error(f"Error processing application {i}: {e}")
                # Create error response
                error_response = LoanPredictionResponse(
                    loan_id=app_data.get('loan_id'),
                    prediction="Error",
                    confidence=0.0,
                    risk_score=1.0,
                    key_factors=[f"Processing error: {str(e)}"],
                    timestamp=datetime.now().isoformat(),
                    model_version=config.get('production.model_version', 'v1.0')
                )
                responses.append(error_response)
        
        logger.info(f"Batch processing completed: {len(responses)} responses generated")
        
        return responses
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded model.
        
        Returns:
            Dictionary with model information
        """
        if self.model is None:
            return {"error": "No model loaded"}
        
        model_info = {
            "model_type": type(self.model).__name__,
            "model_parameters": self.model.get_params(),
            "feature_count": len(self.feature_names) if self.feature_names else "Unknown",
            "feature_names": self.feature_names,
            "model_version": config.get('production.model_version', 'v1.0'),
            "metadata": self.model_metadata
        }
        
        return model_info
    
    def health_check(self) -> Dict[str, Any]:
        """
        Perform health check on the production pipeline.
        
        Returns:
            Health check results
        """
        health_status = {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "checks": {
                "model_loaded": self.model is not None,
                "preprocessor_loaded": self.preprocessor is not None,
                "label_encoder_loaded": self.label_encoder is not None,
                "feature_names_available": self.feature_names is not None
            }
        }
        
        # Overall health
        all_checks_passed = all(health_status["checks"].values())
        health_status["status"] = "healthy" if all_checks_passed else "unhealthy"
        
        if not all_checks_passed:
            health_status["issues"] = [
                check for check, passed in health_status["checks"].items() if not passed
            ]
        
        return health_status


class ModelRegistry:
    """Manages multiple model versions and deployment."""
    
    def __init__(self, registry_path: str = "models/model_registry.json"):
        """
        Initialize model registry.
        
        Args:
            registry_path: Path to model registry file
        """
        self.registry_path = Path(registry_path)
        self.registry_path.parent.mkdir(exist_ok=True)
        self.registry = self._load_registry()
    
    def _load_registry(self) -> Dict[str, Any]:
        """Load model registry from file."""
        if self.registry_path.exists():
            with open(self.registry_path, 'r') as f:
                return json.load(f)
        else:
            return {"models": {}, "active_model": None}
    
    def _save_registry(self) -> None:
        """Save model registry to file."""
        with open(self.registry_path, 'w') as f:
            json.dump(self.registry, f, indent=2, default=str)
    
    def register_model(self, model_name: str, model_path: str, 
                      performance_metrics: Dict[str, float], 
                      metadata: Dict[str, Any]) -> str:
        """
        Register a new model version.
        
        Args:
            model_name: Name of the model
            model_path: Path to the saved model
            performance_metrics: Model performance metrics
            metadata: Additional metadata
            
        Returns:
            Model version ID
        """
        version_id = f"{model_name}_v{len(self.registry['models']) + 1}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        model_entry = {
            "version_id": version_id,
            "model_name": model_name,
            "model_path": model_path,
            "performance_metrics": performance_metrics,
            "metadata": metadata,
            "registration_timestamp": datetime.now().isoformat(),
            "status": "registered"
        }
        
        self.registry["models"][version_id] = model_entry
        self._save_registry()
        
        logger.info(f"Model registered: {version_id}")
        return version_id
    
    def set_active_model(self, version_id: str) -> None:
        """Set a model version as active for production."""
        if version_id not in self.registry["models"]:
            raise ValueError(f"Model version {version_id} not found in registry")
        
        self.registry["active_model"] = version_id
        self.registry["models"][version_id]["status"] = "active"
        
        # Set other models as inactive
        for vid, model_info in self.registry["models"].items():
            if vid != version_id and model_info["status"] == "active":
                model_info["status"] = "inactive"
        
        self._save_registry()
        logger.info(f"Active model set to: {version_id}")
    
    def get_active_model_info(self) -> Optional[Dict[str, Any]]:
        """Get information about the currently active model."""
        active_model_id = self.registry.get("active_model")
        
        if active_model_id and active_model_id in self.registry["models"]:
            return self.registry["models"][active_model_id]
        
        return None
    
    def list_models(self) -> List[Dict[str, Any]]:
        """List all registered models."""
        return list(self.registry["models"].values())


def create_production_pipeline(model_name: str, model_path: str, 
                             performance_metrics: Dict[str, float]) -> ProductionPipeline:
    """
    Create and configure production pipeline with model registration.
    
    Args:
        model_name: Name of the best model
        model_path: Path to the saved model
        performance_metrics: Model performance metrics
        
    Returns:
        Configured ProductionPipeline instance
    """
    # Register model
    registry = ModelRegistry()
    
    metadata = {
        "model_type": model_name,
        "training_timestamp": datetime.now().isoformat(),
        "framework": "scikit-learn",
        "python_version": "3.8+",
        "dependencies": ["scikit-learn", "pandas", "numpy"]
    }
    
    version_id = registry.register_model(model_name, model_path, performance_metrics, metadata)
    registry.set_active_model(version_id)
    
    # Create production pipeline
    pipeline = ProductionPipeline(model_path=model_path)
    
    logger.info("Production pipeline created and configured successfully")
    
    return pipeline
