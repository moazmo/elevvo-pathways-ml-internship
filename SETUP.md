# 🚀 Setup Guide - Elevvo Pathways ML Internship Portfolio

This guide will help you set up and run all projects in this machine learning portfolio.

## 📋 Prerequisites

### System Requirements
- **Python 3.8+** (Python 3.9+ recommended)
- **Git** with Git LFS support
- **Docker** (for containerized projects)
- **Node.js 16+** (for React frontend in Task 4)
- **4GB+ RAM** (8GB+ recommended for deep learning tasks)

### Development Tools (Recommended)
- **VS Code** or **PyCharm** for development
- **Jupyter Lab** for notebook-based projects
- **Postman** for API testing

## 🛠️ Installation Steps

### 1. Clone the Repository
```bash
# Clone with Git LFS support
git lfs clone https://github.com/[your-username]/elevvo-pathways-ml-internship.git
cd elevvo-pathways-ml-internship

# Or if you already cloned without LFS
git lfs pull
```

### 2. Set Up Python Environment
```bash
# Create virtual environment
python -m venv .venv

# Activate virtual environment
# Windows
.venv\Scripts\activate
# Linux/Mac
source .venv/bin/activate

# Upgrade pip
python -m pip install --upgrade pip
```

### 3. Install Global Dependencies (Optional)
```bash
# Install Jupyter for notebook projects
pip install jupyter jupyterlab

# Install common ML libraries
pip install pandas numpy matplotlib seaborn scikit-learn
```

## 📊 Project-Specific Setup

### 🛍️ Task 1: Customer Segmentation
```bash
cd Task_1
pip install -r requirements.txt
jupyter notebook customer_segmentation_analysis.ipynb
```

### 🌲 Task 2: Forest Cover Classification
```bash
cd Task_2
pip install -r requirements.txt
jupyter notebook forest_cover_classification.ipynb
```

### 💰 Task 3: Loan Approval Prediction
```bash
cd Task_3
pip install -r requirements.txt

# Run complete pipeline
python main_pipeline.py

# Start API server
python src/api_server.py

# Run tests
pytest tests/ -v
```

### 🏪 Task 4: Walmart Sales Forecasting
```bash
cd Task_4
pip install -r requirements.txt

# Start full system (includes database, API, and frontend)
python start_full_system.py --frontend

# Or start individual components
python start_api.py  # API only
```

### 🚦 Task 5: Traffic Sign Recognition
```bash
cd Task_5
pip install -r requirements.txt

# Test the application
python test_app.py

# Start web application
cd webapp
python app.py
```

## 🐳 Docker Setup (Alternative)

For projects with Docker support:

```bash
# Task 3: Loan Approval
cd Task_3
docker build -t loan-approval-api .
docker run -p 8000:8000 loan-approval-api

# Task 4: Walmart Sales Forecasting
cd Task_4
docker-compose up --build
```

## 📊 Data Setup

### Large Files with Git LFS
Some projects use Git LFS for large files. Ensure you have:
```bash
# Install Git LFS (if not already installed)
git lfs install

# Pull LFS files
git lfs pull
```

### Sample Data
Each project includes sample data for testing. For full datasets:
- **Task 1**: Mall_Customers.csv (included)
- **Task 2**: Download from UCI ML Repository
- **Task 3**: loan_approval_dataset.csv (included)
- **Task 4**: Walmart sales data (processed version included)
- **Task 5**: GTSRB dataset (download instructions in project README)

## 🧪 Testing

### Run All Tests
```bash
# Task 3: Comprehensive testing
cd Task_3
pytest tests/ -v --cov=src

# Task 4: API testing
cd Task_4
python -m pytest tests/

# Task 5: Application testing
cd Task_5
python test_app.py
```

## 🔧 Troubleshooting

### Common Issues

#### 1. Git LFS Files Not Downloaded
```bash
git lfs pull
```

#### 2. Python Package Conflicts
```bash
# Create fresh environment
python -m venv .venv_new
source .venv_new/bin/activate  # or .venv_new\Scripts\activate on Windows
pip install -r requirements.txt
```

#### 3. Port Already in Use
```bash
# Find and kill process using port (Windows)
netstat -ano | findstr :8000
taskkill /PID <PID> /F

# Linux/Mac
lsof -ti:8000 | xargs kill -9
```

#### 4. Docker Issues
```bash
# Reset Docker
docker system prune -a
docker-compose down
docker-compose up --build
```

#### 5. Node.js/React Issues
```bash
cd Task_4/frontend
rm -rf node_modules package-lock.json
npm install
npm run dev
```

### Memory Issues
For large datasets or models:
```bash
# Increase Python memory limit
export PYTHONHASHSEED=0
export OMP_NUM_THREADS=1

# Use smaller batch sizes in deep learning tasks
# Modify batch_size parameters in config files
```

## 📱 Development Workflow

### 1. Working with Notebooks
```bash
# Start Jupyter Lab
jupyter lab

# Or Jupyter Notebook
jupyter notebook
```

### 2. API Development
```bash
# Task 3 & 4: Auto-reload during development
uvicorn src.api_server:app --reload --host 0.0.0.0 --port 8000
```

### 3. Frontend Development
```bash
# Task 4: React development server
cd Task_4/frontend
npm run dev
```

## 🌐 Accessing Applications

After successful setup:

- **Task 1**: Jupyter notebook at `http://localhost:8888`
- **Task 2**: Jupyter notebook at `http://localhost:8888`
- **Task 3**: API at `http://localhost:8000`, Docs at `http://localhost:8000/docs`
- **Task 4**: Web app at `http://localhost:3000`, API at `http://localhost:8000`
- **Task 5**: Web app at `http://localhost:5000`

## 📞 Support

If you encounter issues:

1. Check the individual project README files
2. Ensure all prerequisites are installed
3. Verify Python and package versions
4. Check the troubleshooting section above
5. Create an issue in the repository

## 🔄 Updates

To get the latest updates:
```bash
git pull origin main
git lfs pull
pip install -r requirements.txt --upgrade
```

---

**Happy coding! 🚀**