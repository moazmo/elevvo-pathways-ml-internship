#!/usr/bin/env python3
"""
Professional deployment script for Loan Approval Web Application.
This script starts the web application with production-ready configuration.
"""

import uvicorn
import sys
import os
from pathlib import Path
import subprocess
import time
import requests
from loguru import logger

def check_dependencies():
    """Check if all required dependencies are installed."""
    required_packages = [
        ('fastapi', 'fastapi'),
        ('uvicorn', 'uvicorn'),
        ('jinja2', 'jinja2'),
        ('python-multipart', 'multipart'),
        ('pandas', 'pandas'),
        ('scikit-learn', 'sklearn'),
        ('joblib', 'joblib'),
        ('loguru', 'loguru')
    ]

    missing_packages = []
    for package_name, import_name in required_packages:
        try:
            __import__(import_name)
        except ImportError:
            missing_packages.append(package_name)
    
    if missing_packages:
        logger.error(f"Missing required packages: {', '.join(missing_packages)}")
        logger.info("Please install missing packages with: pip install " + " ".join(missing_packages))
        return False
    
    return True

def check_model_files():
    """Check if required model files exist."""
    required_files = [
        "models/decision_tree_model.joblib",
        "models/preprocessor.joblib",
        "config/config.yaml"
    ]
    
    missing_files = []
    for file_path in required_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)
    
    if missing_files:
        logger.error(f"Missing required files: {', '.join(missing_files)}")
        logger.info("Please run 'python main_pipeline.py' to generate model files")
        return False
    
    return True

def wait_for_server(url="http://localhost:8000", timeout=30):
    """Wait for the server to be ready."""
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            response = requests.get(f"{url}/health")
            if response.status_code == 200:
                return True
        except requests.exceptions.ConnectionError:
            pass
        time.sleep(1)
    return False

def main():
    """Main deployment function."""
    logger.info("🚀 Starting Loan Approval Web Application Deployment")
    logger.info("=" * 60)
    
    # Check dependencies
    logger.info("📦 Checking dependencies...")
    if not check_dependencies():
        sys.exit(1)
    logger.info("✅ All dependencies satisfied")
    
    # Check model files
    logger.info("🤖 Checking model files...")
    if not check_model_files():
        sys.exit(1)
    logger.info("✅ All model files present")
    
    # Start the web application
    logger.info("🌐 Starting web application server...")
    logger.info("📍 Web App URL: http://localhost:8000")
    logger.info("📊 Admin Dashboard: http://localhost:8000/admin")
    logger.info("📚 API Documentation: http://localhost:8000/docs")
    logger.info("=" * 60)
    logger.info("🎯 Ready for production use!")
    logger.info("Press Ctrl+C to stop the server")
    logger.info("=" * 60)
    
    try:
        # Configure uvicorn for production
        uvicorn.run(
            "src.api_server:app",
            host="0.0.0.0",
            port=8000,
            reload=False,  # Disable reload for production
            access_log=True,
            log_level="info",
            workers=1  # Single worker for local deployment
        )
    except KeyboardInterrupt:
        logger.info("🛑 Server stopped by user")
    except Exception as e:
        logger.error(f"❌ Server error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
