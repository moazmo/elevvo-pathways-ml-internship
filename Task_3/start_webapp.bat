@echo off
echo ================================================================================
echo                    LOAN APPROVAL WEB APPLICATION
echo                         Professional Deployment
echo ================================================================================
echo.

REM Check if virtual environment is activated
if not defined VIRTUAL_ENV (
    echo WARNING: Virtual environment not detected
    echo Please activate your virtual environment first:
    echo   .venv\Scripts\activate
    echo.
    pause
    exit /b 1
)

echo ✅ Virtual environment detected: %VIRTUAL_ENV%
echo.

REM Check if model files exist
if not exist "models\decision_tree_model.joblib" (
    echo ❌ Model files not found!
    echo Please run the training pipeline first:
    echo   python main_pipeline.py
    echo.
    pause
    exit /b 1
)

echo ✅ Model files verified
echo.

REM Install/verify dependencies
echo 📦 Checking dependencies...
pip install fastapi uvicorn jinja2 python-multipart requests --quiet
if %errorlevel% neq 0 (
    echo ❌ Failed to install dependencies
    pause
    exit /b 1
)

echo ✅ Dependencies verified
echo.

REM Start the web application
echo 🚀 Starting Loan Approval Web Application...
echo.
echo 🌐 Web Application: http://localhost:8000
echo 📊 Admin Dashboard: http://localhost:8000/admin  
echo 📚 API Documentation: http://localhost:8000/docs
echo.
echo Press Ctrl+C to stop the server
echo ================================================================================
echo.

python start_webapp.py

echo.
echo ================================================================================
echo                        APPLICATION STOPPED
echo ================================================================================
pause
