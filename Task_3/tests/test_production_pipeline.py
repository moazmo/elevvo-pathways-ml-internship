"""
Unit tests for production pipeline functionality.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import os
import sys
import joblib
from unittest.mock import Mock, patch

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.production_pipeline import (
    LoanApplicationRequest, LoanPredictionResponse, 
    ProductionPipeline, ModelRegistry
)


class TestLoanApplicationRequest:
    """Test suite for LoanApplicationRequest validation."""
    
    def test_valid_application_request(self):
        """Test valid loan application request."""
        valid_data = {
            "loan_id": "TEST_001",
            "no_of_dependents": 2,
            "education": "Graduate",
            "self_employed": "No",
            "income_annum": 5000000.0,
            "loan_amount": 15000000.0,
            "loan_term": 10,
            "cibil_score": 750,
            "residential_assets_value": 2000000.0,
            "commercial_assets_value": 1000000.0,
            "luxury_assets_value": 500000.0,
            "bank_asset_value": 300000.0
        }
        
        request = LoanApplicationRequest(**valid_data)
        assert request.loan_id == "TEST_001"
        assert request.no_of_dependents == 2
        assert request.education == "Graduate"
    
    def test_invalid_education_value(self):
        """Test invalid education value."""
        invalid_data = {
            "no_of_dependents": 2,
            "education": "Invalid Education",  # Invalid value
            "self_employed": "No",
            "income_annum": 5000000.0,
            "loan_amount": 15000000.0,
            "loan_term": 10,
            "cibil_score": 750,
            "residential_assets_value": 2000000.0,
            "commercial_assets_value": 1000000.0,
            "luxury_assets_value": 500000.0,
            "bank_asset_value": 300000.0
        }
        
        with pytest.raises(ValueError):
            LoanApplicationRequest(**invalid_data)
    
    def test_invalid_cibil_score_range(self):
        """Test invalid CIBIL score range."""
        invalid_data = {
            "no_of_dependents": 2,
            "education": "Graduate",
            "self_employed": "No",
            "income_annum": 5000000.0,
            "loan_amount": 15000000.0,
            "loan_term": 10,
            "cibil_score": 1000,  # Invalid range
            "residential_assets_value": 2000000.0,
            "commercial_assets_value": 1000000.0,
            "luxury_assets_value": 500000.0,
            "bank_asset_value": 300000.0
        }
        
        with pytest.raises(ValueError):
            LoanApplicationRequest(**invalid_data)
    
    def test_negative_income(self):
        """Test negative income validation."""
        invalid_data = {
            "no_of_dependents": 2,
            "education": "Graduate",
            "self_employed": "No",
            "income_annum": -1000000.0,  # Negative income
            "loan_amount": 15000000.0,
            "loan_term": 10,
            "cibil_score": 750,
            "residential_assets_value": 2000000.0,
            "commercial_assets_value": 1000000.0,
            "luxury_assets_value": 500000.0,
            "bank_asset_value": 300000.0
        }
        
        with pytest.raises(ValueError):
            LoanApplicationRequest(**invalid_data)


class TestProductionPipeline:
    """Test suite for ProductionPipeline class."""
    
    @pytest.fixture
    def mock_model(self):
        """Create a mock model for testing."""
        model = Mock()
        model.predict.return_value = np.array([1])
        model.predict_proba.return_value = np.array([[0.3, 0.7]])
        model.get_params.return_value = {"test_param": "test_value"}
        return model
    
    @pytest.fixture
    def mock_preprocessor(self):
        """Create a mock preprocessor for testing."""
        preprocessor = Mock()
        preprocessor.transform.return_value = np.array([[1, 2, 3, 4, 5]])
        return preprocessor
    
    @pytest.fixture
    def mock_label_encoder(self):
        """Create a mock label encoder for testing."""
        encoder = Mock()
        encoder.inverse_transform.return_value = ["Approved"]
        return encoder
    
    @pytest.fixture
    def temp_model_files(self, mock_model, mock_preprocessor, mock_label_encoder):
        """Create temporary model files for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create model file
            model_path = Path(temp_dir) / "test_model.joblib"
            model_artifacts = {
                'model': mock_model,
                'metadata': {'test': 'metadata'},
                'feature_names': ['feature1', 'feature2', 'feature3', 'feature4', 'feature5'],
                'model_version': 'v1.0'
            }
            joblib.dump(model_artifacts, model_path)
            
            # Create preprocessor file
            preprocessor_path = Path(temp_dir) / "test_preprocessor.joblib"
            preprocessor_artifacts = {
                'preprocessor': mock_preprocessor,
                'label_encoder': mock_label_encoder,
                'feature_names': ['feature1', 'feature2', 'feature3', 'feature4', 'feature5'],
                'numerical_features': ['feature1', 'feature2'],
                'categorical_features': ['feature3', 'feature4']
            }
            joblib.dump(preprocessor_artifacts, preprocessor_path)
            
            yield str(model_path), str(preprocessor_path)
    
    def test_pipeline_initialization_success(self, temp_model_files):
        """Test successful pipeline initialization."""
        model_path, preprocessor_path = temp_model_files
        
        pipeline = ProductionPipeline(model_path=model_path, preprocessor_path=preprocessor_path)
        
        assert pipeline.model is not None
        assert pipeline.preprocessor is not None
        assert pipeline.label_encoder is not None
        assert pipeline.feature_names is not None
    
    def test_pipeline_initialization_missing_files(self):
        """Test pipeline initialization with missing files."""
        # This should not raise an error but should log warnings
        pipeline = ProductionPipeline(
            model_path="non_existent_model.joblib",
            preprocessor_path="non_existent_preprocessor.joblib"
        )
        
        assert pipeline.model is None
        assert pipeline.preprocessor is None
    
    @patch('src.production_pipeline.config')
    def test_validate_input_success(self, mock_config, temp_model_files):
        """Test successful input validation."""
        model_path, preprocessor_path = temp_model_files
        
        # Mock config
        mock_config.get.return_value = {
            'numerical_features': ['feature1', 'feature2'],
            'categorical_features': ['feature3', 'feature4']
        }
        
        pipeline = ProductionPipeline(model_path=model_path, preprocessor_path=preprocessor_path)
        
        valid_data = {
            "no_of_dependents": 2,
            "education": "Graduate",
            "self_employed": "No",
            "income_annum": 5000000.0,
            "loan_amount": 15000000.0,
            "loan_term": 10,
            "cibil_score": 750,
            "residential_assets_value": 2000000.0,
            "commercial_assets_value": 1000000.0,
            "luxury_assets_value": 500000.0,
            "bank_asset_value": 300000.0
        }
        
        validated_data = pipeline._validate_input(valid_data)
        assert isinstance(validated_data, LoanApplicationRequest)
    
    def test_validate_input_failure(self, temp_model_files):
        """Test input validation failure."""
        model_path, preprocessor_path = temp_model_files
        
        pipeline = ProductionPipeline(model_path=model_path, preprocessor_path=preprocessor_path)
        
        invalid_data = {
            "education": "Invalid Education",  # Invalid value
            "income_annum": -1000000.0  # Negative income
        }
        
        with pytest.raises(ValueError):
            pipeline._validate_input(invalid_data)
    
    @patch('src.production_pipeline.config')
    def test_predict_success(self, mock_config, temp_model_files):
        """Test successful prediction."""
        model_path, preprocessor_path = temp_model_files
        
        # Mock config
        mock_config.get.return_value = {
            'numerical_features': ['income_annum', 'loan_amount', 'loan_term', 'cibil_score',
                                 'residential_assets_value', 'commercial_assets_value',
                                 'luxury_assets_value', 'bank_asset_value', 'no_of_dependents'],
            'categorical_features': ['education', 'self_employed']
        }
        
        pipeline = ProductionPipeline(model_path=model_path, preprocessor_path=preprocessor_path)
        
        application_data = {
            "loan_id": "TEST_001",
            "no_of_dependents": 2,
            "education": "Graduate",
            "self_employed": "No",
            "income_annum": 5000000.0,
            "loan_amount": 15000000.0,
            "loan_term": 10,
            "cibil_score": 750,
            "residential_assets_value": 2000000.0,
            "commercial_assets_value": 1000000.0,
            "luxury_assets_value": 500000.0,
            "bank_asset_value": 300000.0
        }
        
        response = pipeline.predict(application_data)
        
        assert isinstance(response, LoanPredictionResponse)
        assert response.loan_id == "TEST_001"
        assert response.prediction in ["Approved", "Rejected"]
        assert 0 <= response.confidence <= 1
        assert 0 <= response.risk_score <= 1
        assert isinstance(response.key_factors, list)
    
    def test_predict_no_model_loaded(self):
        """Test prediction with no model loaded."""
        pipeline = ProductionPipeline(
            model_path="non_existent_model.joblib",
            preprocessor_path="non_existent_preprocessor.joblib"
        )
        
        application_data = {
            "no_of_dependents": 2,
            "education": "Graduate",
            "self_employed": "No",
            "income_annum": 5000000.0,
            "loan_amount": 15000000.0,
            "loan_term": 10,
            "cibil_score": 750,
            "residential_assets_value": 2000000.0,
            "commercial_assets_value": 1000000.0,
            "luxury_assets_value": 500000.0,
            "bank_asset_value": 300000.0
        }
        
        with pytest.raises(ValueError, match="Model not loaded"):
            pipeline.predict(application_data)
    
    def test_health_check_healthy(self, temp_model_files):
        """Test health check with healthy pipeline."""
        model_path, preprocessor_path = temp_model_files
        
        pipeline = ProductionPipeline(model_path=model_path, preprocessor_path=preprocessor_path)
        
        health_status = pipeline.health_check()
        
        assert health_status["status"] == "healthy"
        assert health_status["checks"]["model_loaded"] is True
        assert health_status["checks"]["preprocessor_loaded"] is True
        assert health_status["checks"]["label_encoder_loaded"] is True
        assert health_status["checks"]["feature_names_available"] is True
    
    def test_health_check_unhealthy(self):
        """Test health check with unhealthy pipeline."""
        pipeline = ProductionPipeline(
            model_path="non_existent_model.joblib",
            preprocessor_path="non_existent_preprocessor.joblib"
        )
        
        health_status = pipeline.health_check()
        
        assert health_status["status"] == "unhealthy"
        assert health_status["checks"]["model_loaded"] is False
        assert health_status["checks"]["preprocessor_loaded"] is False
        assert "issues" in health_status


class TestModelRegistry:
    """Test suite for ModelRegistry class."""
    
    @pytest.fixture
    def temp_registry_file(self):
        """Create temporary registry file for testing."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            yield f.name
        os.unlink(f.name)
    
    def test_registry_initialization_new_file(self, temp_registry_file):
        """Test registry initialization with new file."""
        registry = ModelRegistry(registry_path=temp_registry_file)
        
        assert registry.registry["models"] == {}
        assert registry.registry["active_model"] is None
    
    def test_register_model(self, temp_registry_file):
        """Test model registration."""
        registry = ModelRegistry(registry_path=temp_registry_file)
        
        performance_metrics = {"accuracy": 0.85, "f1_score": 0.82}
        metadata = {"model_type": "logistic_regression"}
        
        version_id = registry.register_model(
            model_name="test_model",
            model_path="/path/to/model.joblib",
            performance_metrics=performance_metrics,
            metadata=metadata
        )
        
        assert version_id in registry.registry["models"]
        assert registry.registry["models"][version_id]["model_name"] == "test_model"
        assert registry.registry["models"][version_id]["performance_metrics"] == performance_metrics
    
    def test_set_active_model(self, temp_registry_file):
        """Test setting active model."""
        registry = ModelRegistry(registry_path=temp_registry_file)
        
        # Register a model first
        version_id = registry.register_model(
            model_name="test_model",
            model_path="/path/to/model.joblib",
            performance_metrics={"accuracy": 0.85},
            metadata={}
        )
        
        # Set as active
        registry.set_active_model(version_id)
        
        assert registry.registry["active_model"] == version_id
        assert registry.registry["models"][version_id]["status"] == "active"
    
    def test_set_active_model_not_found(self, temp_registry_file):
        """Test setting active model with non-existent version."""
        registry = ModelRegistry(registry_path=temp_registry_file)
        
        with pytest.raises(ValueError, match="Model version .* not found"):
            registry.set_active_model("non_existent_version")
    
    def test_get_active_model_info(self, temp_registry_file):
        """Test getting active model information."""
        registry = ModelRegistry(registry_path=temp_registry_file)
        
        # No active model initially
        assert registry.get_active_model_info() is None
        
        # Register and set active model
        version_id = registry.register_model(
            model_name="test_model",
            model_path="/path/to/model.joblib",
            performance_metrics={"accuracy": 0.85},
            metadata={}
        )
        registry.set_active_model(version_id)
        
        active_info = registry.get_active_model_info()
        assert active_info is not None
        assert active_info["version_id"] == version_id
        assert active_info["model_name"] == "test_model"


if __name__ == "__main__":
    pytest.main([__file__])
