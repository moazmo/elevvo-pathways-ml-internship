"""
Unit tests for data loading functionality.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import os
import sys

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.data_loader import DataLoader


class TestDataLoader:
    """Test suite for DataLoader class."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample loan data for testing."""
        data = {
            'loan_id': [1, 2, 3, 4, 5],
            'no_of_dependents': [2, 0, 3, 1, 2],
            'education': ['Graduate', 'Not Graduate', 'Graduate', 'Graduate', 'Not Graduate'],
            'self_employed': ['No', 'Yes', 'No', 'Yes', 'No'],
            'income_annum': [5000000, 3000000, 7000000, 4500000, 6000000],
            'loan_amount': [15000000, 8000000, 20000000, 12000000, 18000000],
            'loan_term': [10, 15, 8, 12, 20],
            'cibil_score': [750, 650, 800, 700, 720],
            'residential_assets_value': [2000000, 1500000, 3000000, 2500000, 2200000],
            'commercial_assets_value': [1000000, 500000, 2000000, 1200000, 1500000],
            'luxury_assets_value': [500000, 200000, 1000000, 600000, 800000],
            'bank_asset_value': [300000, 100000, 500000, 400000, 350000],
            'loan_status': ['Approved', 'Rejected', 'Approved', 'Rejected', 'Approved']
        }
        return pd.DataFrame(data)
    
    @pytest.fixture
    def temp_csv_file(self, sample_data):
        """Create temporary CSV file for testing."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            sample_data.to_csv(f.name, index=False)
            yield f.name
        os.unlink(f.name)
    
    def test_data_loader_initialization(self):
        """Test DataLoader initialization."""
        loader = DataLoader()
        assert loader.data_config is not None
        assert loader.feature_config is not None
    
    def test_load_raw_data_success(self, temp_csv_file, monkeypatch):
        """Test successful data loading."""
        # Mock the config to use our temp file
        def mock_get_data_config():
            return {'raw_data_path': temp_csv_file}
        
        monkeypatch.setattr('src.data_loader.config.get_data_config', mock_get_data_config)
        
        loader = DataLoader()
        df = loader.load_raw_data()
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 5
        assert 'loan_status' in df.columns
    
    def test_load_raw_data_file_not_found(self, monkeypatch):
        """Test data loading with non-existent file."""
        def mock_get_data_config():
            return {'raw_data_path': 'non_existent_file.csv'}
        
        monkeypatch.setattr('src.data_loader.config.get_data_config', mock_get_data_config)
        
        loader = DataLoader()
        
        with pytest.raises(FileNotFoundError):
            loader.load_raw_data()
    
    def test_validate_data_schema_success(self, sample_data):
        """Test successful data schema validation."""
        loader = DataLoader()
        result = loader.validate_data_schema(sample_data)
        assert result is True
    
    def test_validate_data_schema_missing_columns(self, sample_data):
        """Test data schema validation with missing columns."""
        # Remove a required column
        incomplete_data = sample_data.drop(columns=['cibil_score'])
        
        loader = DataLoader()
        result = loader.validate_data_schema(incomplete_data)
        assert result is False
    
    def test_validate_data_schema_invalid_target_values(self, sample_data):
        """Test data schema validation with invalid target values."""
        # Change target values to invalid ones
        invalid_data = sample_data.copy()
        invalid_data['loan_status'] = ['Valid', 'Invalid', 'Other', 'Values', 'Here']
        
        loader = DataLoader()
        result = loader.validate_data_schema(invalid_data)
        assert result is False
    
    def test_get_data_summary(self, sample_data):
        """Test data summary generation."""
        loader = DataLoader()
        summary = loader.get_data_summary(sample_data)
        
        assert 'shape' in summary
        assert 'columns' in summary
        assert 'missing_values' in summary
        assert 'target_distribution' in summary
        assert 'numerical_summary' in summary
        assert 'categorical_summary' in summary
        
        # Check specific values
        assert summary['shape'] == (5, 13)
        assert summary['target_distribution']['Approved'] == 3
        assert summary['target_distribution']['Rejected'] == 2
    
    def test_load_and_validate_integration(self, temp_csv_file, monkeypatch):
        """Test the complete load and validate workflow."""
        def mock_get_data_config():
            return {'raw_data_path': temp_csv_file}
        
        monkeypatch.setattr('src.data_loader.config.get_data_config', mock_get_data_config)
        
        loader = DataLoader()
        df, summary = loader.load_and_validate()
        
        assert isinstance(df, pd.DataFrame)
        assert isinstance(summary, dict)
        assert len(df) == 5
        assert 'target_distribution' in summary


if __name__ == "__main__":
    pytest.main([__file__])
