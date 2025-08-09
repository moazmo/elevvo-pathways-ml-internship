# 💰 Loan Approval Prediction System

> **Part of Elevvo Pathways ML Internship Portfolio**  
> *Production-ready ML system with comprehensive MLOps pipeline*

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/API-FastAPI-green.svg)](https://fastapi.tiangolo.com)
[![Docker](https://img.shields.io/badge/Deploy-Docker-blue.svg)](https://docker.com)
[![Tests](https://img.shields.io/badge/Tests-pytest-yellow.svg)](https://pytest.org)

A professional-grade machine learning system for predicting loan approval decisions with production-ready deployment capabilities.

## 🎯 Project Overview

This project implements a comprehensive binary classification system to predict loan approval decisions, with special focus on:

- **Imbalanced Data Handling**: Using SMOTE and advanced resampling techniques
- **Model Comparison**: Logistic Regression vs Decision Tree vs Ensemble Methods
- **Business-Focused Metrics**: Precision, Recall, F1-Score with emphasis on financial risk
- **Production Readiness**: Complete API deployment with monitoring and validation

## 📊 Dataset

- **Source**: Loan Approval Prediction Dataset (Kaggle-style)
- **Size**: ~4,270 loan applications
- **Features**: 11 predictive features including demographics, financial, and credit information
- **Target**: Binary classification (Approved/Rejected)

### Features:
- **Categorical**: education, self_employed
- **Numerical**: no_of_dependents, income_annum, loan_amount, loan_term, cibil_score, various asset values

## 🏗️ Project Structure

```
Task_3/
├── loan_approval_dataset.csv          # Raw dataset
├── loan_approval_analysis.ipynb       # Main analysis notebook
├── main_pipeline.py                   # Complete pipeline execution
├── requirements.txt                   # Python dependencies
├── README.md                          # This file
├── config/
│   └── config.yaml                    # Configuration settings
├── src/                               # Source code modules
│   ├── __init__.py
│   ├── config_loader.py              # Configuration management
│   ├── data_loader.py                # Data loading and validation
│   ├── eda_analyzer.py               # Exploratory data analysis
│   ├── visualizer.py                 # EDA visualizations
│   ├── data_preprocessor.py          # Data preprocessing pipeline
│   ├── imbalance_handler.py          # Class imbalance handling
│   ├── model_trainer.py              # Model training and tuning
│   ├── model_evaluator.py            # Model evaluation
│   ├── evaluation_visualizer.py      # Evaluation visualizations
│   ├── production_pipeline.py        # Production inference pipeline
│   └── api_server.py                 # FastAPI production server
├── tests/                            # Unit tests
│   ├── __init__.py
│   ├── conftest.py                   # Test configuration
│   ├── test_data_loader.py
│   └── test_production_pipeline.py
├── models/                           # Saved models and artifacts
├── results/                          # Analysis results and visualizations
├── logs/                            # Application logs
└── config/                          # Configuration files
```

## 🚀 Quick Start

### 1. Environment Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Verify installation
python -c "import sklearn, pandas, numpy; print('✅ Dependencies installed')"
```

### 2. Run Complete Pipeline

```bash
# Execute the complete ML pipeline
python main_pipeline.py
```

### 3. Interactive Analysis

```bash
# Launch Jupyter notebook for interactive analysis
jupyter notebook loan_approval_analysis.ipynb
```

### 4. Production API

```bash
# Start the production API server
python src/api_server.py

# API will be available at: http://localhost:8000
# Documentation at: http://localhost:8000/docs
```

## 📈 Key Features

### 🔍 Comprehensive EDA
- Missing value analysis and handling
- Feature distribution analysis
- Target variable imbalance assessment
- Correlation analysis and multicollinearity detection

### ⚖️ Advanced Imbalance Handling
- **SMOTE**: Synthetic Minority Oversampling Technique
- **ADASYN**: Adaptive Synthetic Sampling
- **SMOTE + Tomek**: Combined over/under sampling
- Automatic method selection based on data characteristics

### 🤖 Multiple Model Comparison
- **Logistic Regression**: Interpretable linear model
- **Decision Tree**: Non-linear, rule-based model
- **Random Forest**: Ensemble method for robust predictions
- **Gradient Boosting**: Advanced ensemble technique

### 📊 Business-Focused Evaluation
- **Precision**: Minimize bad loan approvals (Type I errors)
- **Recall**: Capture good loan applications (avoid Type II errors)
- **F1-Score**: Balanced metric for imbalanced data
- **ROC-AUC**: Overall discriminative ability
- **Business Impact**: Cost analysis and risk assessment

### 🏭 Production-Ready Features
- **FastAPI Service**: RESTful API with automatic documentation
- **Input Validation**: Pydantic models for data validation
- **Model Registry**: Version management and deployment tracking
- **Health Monitoring**: System health checks and status monitoring
- **Batch Processing**: Support for multiple predictions
- **Error Handling**: Comprehensive error management

## 📋 API Endpoints

### Core Prediction Endpoints
- `POST /predict` - Single loan application prediction
- `POST /predict/batch` - Batch loan application predictions

### Management Endpoints
- `GET /health` - System health check
- `GET /model/info` - Current model information
- `GET /models/list` - List all registered models
- `GET /docs` - Interactive API documentation

### Example API Usage

```python
import requests

# Single prediction
application_data = {
    "loan_id": "APP_001",
    "no_of_dependents": 2,
    "education": "Graduate",
    "self_employed": "No",
    "income_annum": 6000000,
    "loan_amount": 18000000,
    "loan_term": 15,
    "cibil_score": 780,
    "residential_assets_value": 2500000,
    "commercial_assets_value": 1200000,
    "luxury_assets_value": 600000,
    "bank_asset_value": 400000
}

response = requests.post("http://localhost:8000/predict", json=application_data)
prediction = response.json()

print(f"Prediction: {prediction['prediction']}")
print(f"Confidence: {prediction['confidence']}")
print(f"Risk Score: {prediction['risk_score']}")
```

## 🧪 Testing

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Run specific test file
pytest tests/test_production_pipeline.py -v
```

## 📊 Results and Performance

The system achieves:
- **High Precision**: Minimizes costly bad loan approvals
- **Balanced Recall**: Captures most good loan applications
- **Robust Performance**: Handles class imbalance effectively
- **Production Ready**: Scalable API with comprehensive monitoring

## 🔧 Configuration

All system parameters are configurable via `config/config.yaml`:

- **Data Processing**: Train/test splits, feature selection
- **Model Parameters**: Hyperparameter grids for tuning
- **Imbalance Handling**: SMOTE parameters and method selection
- **Evaluation**: Metrics and cross-validation settings
- **Production**: API settings and deployment configuration

## 📝 Logging and Monitoring

- **Structured Logging**: Comprehensive logging with Loguru
- **Performance Tracking**: Model performance monitoring
- **Error Handling**: Graceful error management and reporting
- **Health Checks**: System status monitoring

## 🚀 Deployment

### Local Development
```bash
python src/api_server.py
```

### Production Deployment
1. Configure production settings in `config/config.yaml`
2. Set up proper logging and monitoring
3. Deploy using Docker or cloud services
4. Implement load balancing and scaling as needed

## 📚 Dependencies

Key libraries used:
- **scikit-learn**: Machine learning algorithms
- **imbalanced-learn**: Class imbalance handling
- **pandas/numpy**: Data manipulation
- **matplotlib/seaborn**: Visualization
- **FastAPI**: Production API framework
- **pydantic**: Data validation
- **pytest**: Testing framework

## 🤝 Contributing

1. Follow the existing code structure and patterns
2. Add comprehensive tests for new features
3. Update documentation and configuration as needed
4. Ensure all tests pass before submitting changes

## 📄 License

This project is developed for educational and professional demonstration purposes.

---

**Built with ❤️ for professional machine learning deployment**
