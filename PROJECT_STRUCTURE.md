# 📁 Project Structure - Elevvo Pathways ML Internship

This document outlines the complete structure of the machine learning internship portfolio.

## 🏗️ Repository Structure

```
elevvo-pathways-ml-internship/
├── 📋 README.md                    # Main portfolio documentation
├── 🚀 SETUP.md                     # Comprehensive setup guide
├── 🤝 CONTRIBUTING.md              # Contribution guidelines
├── 📁 PROJECT_STRUCTURE.md         # This file
├── 📄 LICENSE                      # MIT License
├── 🔒 .gitignore                   # Git ignore rules
├── 📦 .gitattributes               # Git LFS configuration
├── 🛠️ requirements-dev.txt         # Development dependencies
├── 🔄 .github/                     # GitHub Actions workflows
│   └── workflows/
│       └── ci.yml                  # CI/CD pipeline
├── 🛍️ Task_1/                      # Customer Segmentation Analysis
├── 🌲 Task_2/                      # Forest Cover Classification
├── 💰 Task_3/                      # Loan Approval Prediction System
├── 🏪 Task_4/                      # Walmart Sales Forecasting System
└── 🚦 Task_5/                      # Traffic Sign Recognition
```

## 📊 Individual Project Structures

### 🛍️ Task 1: Customer Segmentation Analysis
```
Task_1/
├── 📋 README.md                           # Project documentation
├── 📊 Mall_Customers.csv                  # Dataset (Git LFS)
├── 📓 customer_segmentation_analysis.ipynb # Main analysis notebook
├── 📦 requirements.txt                    # Python dependencies
└── 📈 results/                           # Analysis outputs (gitignored)
    ├── visualizations/
    ├── cluster_analysis/
    └── business_insights/
```

### 🌲 Task 2: Forest Cover Classification
```
Task_2/
├── 📋 README.md                          # Project documentation
├── 🗜️ covtype.data.gz                    # Compressed dataset (Git LFS)
├── 📄 covtype.info                       # Dataset documentation
├── 📓 forest_cover_classification.ipynb  # Main analysis notebook
├── 📦 requirements.txt                   # Python dependencies
└── 📈 results/                          # Analysis outputs (gitignored)
    ├── models/
    ├── evaluations/
    └── feature_importance/
```

### 💰 Task 3: Loan Approval Prediction System
```
Task_3/
├── 📋 README.md                     # Project documentation
├── 📊 loan_approval_dataset.csv     # Dataset (Git LFS)
├── 📓 loan_approval_analysis.ipynb  # Analysis notebook
├── 🚀 main_pipeline.py              # Complete pipeline execution
├── 📦 requirements.txt              # Python dependencies
├── ⚙️ config/
│   └── config.yaml                  # Configuration settings
├── 🐍 src/                          # Source code modules
│   ├── __init__.py
│   ├── config_loader.py             # Configuration management
│   ├── data_loader.py               # Data loading and validation
│   ├── eda_analyzer.py              # Exploratory data analysis
│   ├── visualizer.py                # EDA visualizations
│   ├── data_preprocessor.py         # Data preprocessing pipeline
│   ├── imbalance_handler.py         # Class imbalance handling
│   ├── model_trainer.py             # Model training and tuning
│   ├── model_evaluator.py           # Model evaluation
│   ├── evaluation_visualizer.py     # Evaluation visualizations
│   ├── production_pipeline.py       # Production inference pipeline
│   └── api_server.py                # FastAPI production server
├── 🧪 tests/                        # Unit tests
│   ├── __init__.py
│   ├── conftest.py                  # Test configuration
│   ├── test_data_loader.py
│   └── test_production_pipeline.py
├── 🤖 models/                       # Saved models (gitignored)
├── 📈 results/                      # Analysis results (gitignored)
├── 📝 logs/                         # Application logs (gitignored)
└── ⚙️ config/                       # Configuration files
```

### 🏪 Task 4: Walmart Sales Forecasting System
```
Task_4/
├── 📋 README.md                     # Project documentation
├── 🚀 start_full_system.py          # System launcher
├── 🚀 start_api.py                  # API server launcher
├── 📦 requirements.txt              # Python dependencies
├── 🐳 docker-compose.yml            # Multi-service orchestration
├── 🐳 Dockerfile                    # Docker configuration
├── 🎨 frontend/                     # React web application
│   ├── src/
│   │   ├── components/              # React components
│   │   ├── services/                # API integration
│   │   └── App.jsx                  # Main application
│   ├── package.json
│   └── Dockerfile
├── 🗄️ database/                     # Database configuration
│   └── init.sql                     # Database schema
├── 🐍 src/                          # Python backend
│   ├── api_server.py                # FastAPI application
│   ├── database/                    # Database models & connection
│   ├── data/                        # Data loading utilities
│   └── utils/                       # Configuration & logging
├── 📊 data/                         # Training data (Git LFS)
│   ├── raw/
│   └── processed/
├── 🤖 results/                      # Trained models (Git LFS)
├── 📓 notebooks/                    # Jupyter notebooks
├── 🧪 tests/                        # Test suite
└── 📝 logs/                         # Application logs (gitignored)
```

### 🚦 Task 5: Traffic Sign Recognition
```
Task_5/
├── 📋 README.md                     # Project documentation
├── 🧪 test_app.py                   # Application test script
├── 📦 requirements.txt              # Python dependencies
├── 🐳 Dockerfile                    # Docker configuration
├── 📓 notebooks/                    # Jupyter notebooks for ML pipeline
│   ├── 01_data_exploration.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_model_training.ipynb
│   └── 04_model_comparison.ipynb
├── 🧠 src/                          # Source code modules
│   ├── model_loader.py              # Model loading and inference
│   └── utils.py                     # Utility functions
├── 🌐 webapp/                       # Flask web application
│   ├── app.py                       # Main Flask app
│   ├── templates/                   # HTML templates
│   └── static/                      # CSS, JS, uploads
├── 📁 data/                         # Dataset (download required)
│   ├── raw/                         # Original GTSRB data
│   └── processed/                   # Preprocessed data
├── 🎯 models/                       # Trained models (Git LFS)
├── 🚀 production/                   # Production model (Git LFS)
└── 🖼️ images/                       # Screenshots and documentation
```

## 📦 File Types and Git LFS Configuration

### Git LFS Tracked Files
Large files are tracked with Git LFS to keep the repository lightweight:

- **Data Files**: `*.csv`, `*.json`, `*.parquet`, `*.h5`
- **Model Files**: `*.pkl`, `*.joblib`, `*.pt`, `*.pth`, `*.model`
- **Binary Files**: `*.data`, `*.bin`, `*.weights`
- **Media Files**: `*.jpg`, `*.png`, `*.mp4`, `*.avi`
- **Compressed Files**: `*.zip`, `*.tar.gz`, `*.7z`

### Gitignored Directories
The following directories are excluded from version control:

- **Virtual Environments**: `.venv/`, `venv/`, `.venv*/`
- **Cache Directories**: `__pycache__/`, `.pytest_cache/`, `.cache/`
- **Results**: `results/`, `output/`, `outputs/`
- **Logs**: `logs/`, `*.log`
- **Node Modules**: `node_modules/`
- **Build Artifacts**: `build/`, `dist/`

## 🔧 Configuration Files

### Repository Level
- **`.gitignore`**: Comprehensive ignore rules for all project types
- **`.gitattributes`**: Git LFS configuration for large files
- **`requirements-dev.txt`**: Development dependencies
- **`.github/workflows/ci.yml`**: CI/CD pipeline configuration

### Project Level
- **`requirements.txt`**: Project-specific Python dependencies
- **`config.yaml`**: Configuration settings (Task 3)
- **`docker-compose.yml`**: Multi-service orchestration (Task 4)
- **`package.json`**: Node.js dependencies (Task 4 frontend)

## 📊 Data Management Strategy

### Small Files (< 10MB)
- Included directly in the repository
- Examples: configuration files, small datasets, documentation

### Medium Files (10MB - 100MB)
- Tracked with Git LFS
- Examples: processed datasets, trained models

### Large Files (> 100MB)
- Tracked with Git LFS
- Download instructions provided in project READMEs
- Examples: raw datasets, large pre-trained models

### Sensitive Data
- Never committed to the repository
- Use environment variables for configuration
- Provide sample/dummy data for testing

## 🧪 Testing Strategy

### Unit Tests
- Located in `tests/` directories
- Use pytest framework
- Aim for 80%+ code coverage

### Integration Tests
- Test API endpoints and workflows
- Use Docker for consistent environments
- Automated in CI/CD pipeline

### Manual Testing
- Application testing scripts provided
- User acceptance testing for web applications
- Performance testing for ML models

## 📚 Documentation Standards

### README Files
- Comprehensive project descriptions
- Clear setup and usage instructions
- Performance metrics and business impact
- Professional formatting with badges

### Code Documentation
- Docstrings for all functions and classes
- Type hints for better code understanding
- Inline comments for complex logic

### API Documentation
- OpenAPI/Swagger for REST APIs
- Interactive documentation endpoints
- Request/response examples

## 🚀 Deployment Architecture

### Development
- Local development with virtual environments
- Jupyter notebooks for experimentation
- Hot-reload for web applications

### Testing
- Automated testing in CI/CD pipeline
- Docker containers for consistency
- Multiple Python version testing

### Production
- Docker containerization
- API deployment with monitoring
- Database integration where applicable

---

This structure demonstrates professional software development practices and provides a solid foundation for machine learning project development and deployment.

**Built with ❤️ during Elevvo Pathways internship**