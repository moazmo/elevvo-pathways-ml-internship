"""
Pytest configuration and shared fixtures.
"""

import pytest
import pandas as pd
import numpy as np
import tempfile
import os
from pathlib import Path


@pytest.fixture(scope="session")
def sample_loan_dataset():
    """Create a comprehensive sample loan dataset for testing."""
    np.random.seed(42)
    n_samples = 100
    
    data = {
        'loan_id': range(1, n_samples + 1),
        'no_of_dependents': np.random.randint(0, 6, n_samples),
        'education': np.random.choice(['Graduate', 'Not Graduate'], n_samples),
        'self_employed': np.random.choice(['Yes', 'No'], n_samples),
        'income_annum': np.random.normal(5000000, 2000000, n_samples).clip(min=500000),
        'loan_amount': np.random.normal(15000000, 8000000, n_samples).clip(min=1000000),
        'loan_term': np.random.randint(5, 25, n_samples),
        'cibil_score': np.random.randint(300, 900, n_samples),
        'residential_assets_value': np.random.normal(2000000, 1000000, n_samples).clip(min=0),
        'commercial_assets_value': np.random.normal(1000000, 800000, n_samples).clip(min=0),
        'luxury_assets_value': np.random.normal(500000, 400000, n_samples).clip(min=0),
        'bank_asset_value': np.random.normal(300000, 200000, n_samples).clip(min=0)
    }
    
    # Create realistic loan_status based on some rules
    loan_status = []
    for i in range(n_samples):
        # Simple rule-based approval logic for testing
        score = 0
        if data['cibil_score'][i] > 700:
            score += 2
        if data['income_annum'][i] > 4000000:
            score += 1
        if data['loan_amount'][i] / data['income_annum'][i] < 4:  # Reasonable debt-to-income
            score += 1
        if data['education'][i] == 'Graduate':
            score += 1
        
        # Add some randomness
        if np.random.random() > 0.2:  # 80% follow the rules
            loan_status.append('Approved' if score >= 3 else 'Rejected')
        else:  # 20% random
            loan_status.append(np.random.choice(['Approved', 'Rejected']))
    
    data['loan_status'] = loan_status
    
    return pd.DataFrame(data)


@pytest.fixture
def temp_csv_file(sample_loan_dataset):
    """Create temporary CSV file with sample data."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        sample_loan_dataset.to_csv(f.name, index=False)
        yield f.name
    os.unlink(f.name)


@pytest.fixture
def sample_application_data():
    """Sample loan application data for testing."""
    return {
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


@pytest.fixture
def sample_batch_applications():
    """Sample batch of loan applications for testing."""
    base_data = {
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
    
    applications = []
    for i in range(5):
        app_data = base_data.copy()
        app_data["loan_id"] = f"TEST_{i+1:03d}"
        app_data["cibil_score"] = 700 + i * 20  # Vary credit scores
        applications.append(app_data)
    
    return applications


@pytest.fixture(autouse=True)
def setup_test_environment():
    """Set up test environment before each test."""
    # Ensure test directories exist
    test_dirs = ['models', 'results', 'logs', 'config']
    for dir_name in test_dirs:
        Path(dir_name).mkdir(exist_ok=True)
    
    yield
    
    # Cleanup can be added here if needed
