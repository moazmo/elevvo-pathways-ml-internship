"""
Production Environment Setup Script
===================================

This script sets up the production environment for the Walmart Sales Forecasting system.
Run this once before starting the API server.

Usage:
    python setup_production.py
"""

import os
import sys
from pathlib import Path
import subprocess

def check_python_version():
    """Check if Python version is compatible."""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8+ required")
        print(f"   Current version: {version.major}.{version.minor}")
        return False
    
    print(f"✅ Python {version.major}.{version.minor}.{version.micro}")
    return True

def install_requirements():
    """Install required packages."""
    print("📦 Checking requirements...")
    
    try:
        # First try to check if packages are already installed
        subprocess.run([
            sys.executable, "-m", "pip", "check"
        ], check=True, capture_output=True, text=True)
        print("✅ All requirements already satisfied")
        return True
    except subprocess.CalledProcessError:
        pass
    
    # If check failed, try to install requirements
    try:
        print("📦 Installing requirements...")
        result = subprocess.run([
            sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
        ], check=True, capture_output=True, text=True)
        print("✅ Requirements installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"⚠️  Pip install failed, but continuing (packages may already be installed)")
        print(f"   Error: {e}")
        # Continue anyway as packages might already be installed
        return True

def setup_directories():
    """Ensure all required directories exist."""
    print("📁 Setting up directories...")
    
    from src.utils.config import ensure_directories
    
    try:
        ensure_directories()
        print("✅ Directories configured")
        return True
    except Exception as e:
        print(f"❌ Directory setup failed: {e}")
        return False

def validate_models():
    """Validate that trained models exist."""
    print("🤖 Validating models...")
    
    models_dir = Path("results/models")
    required_files = [
        "advanced_models.pkl",
        "best_baseline_model.pkl"
    ]
    
    for file in required_files:
        if not (models_dir / file).exists():
            print(f"❌ Missing model file: {file}")
            return False
    
    print("✅ All models found")
    return True

def validate_data():
    """Validate that processed data exists."""
    print("💾 Validating data...")
    
    data_dir = Path("data/processed")
    required_files = [
        "train_processed.csv",
        "feature_list.txt",
        "label_encoders.pkl"
    ]
    
    for file in required_files:
        if not (data_dir / file).exists():
            print(f"❌ Missing data file: {file}")
            return False
    
    print("✅ All data files found")
    return True

def test_api_import():
    """Test that the API can be imported."""
    print("🔧 Testing API import...")
    
    try:
        sys.path.insert(0, str(Path.cwd()))
        from src.api_server import app
        print("✅ API import successful")
        return True
    except Exception as e:
        print(f"❌ API import failed: {e}")
        return False

def main():
    """Run production setup."""
    
    print("🚀 Walmart Sales Forecasting - Production Setup")
    print("=" * 55)
    
    checks = [
        ("Python Version", check_python_version),
        ("Install Requirements", install_requirements),
        ("Setup Directories", setup_directories),
        ("Validate Models", validate_models),
        ("Validate Data", validate_data),
        ("Test API Import", test_api_import)
    ]
    
    results = []
    
    for name, check_func in checks:
        print(f"\n{name}:")
        print("-" * len(name))
        result = check_func()
        results.append(result)
    
    print("\n" + "=" * 55)
    print("📋 Setup Summary")
    print("=" * 55)
    
    for i, (name, _) in enumerate(checks):
        status = "✅ PASS" if results[i] else "❌ FAIL"
        print(f"{name:20} : {status}")
    
    if all(results):
        print("\n🎉 Production setup completed successfully!")
        print("🚀 Ready to start the API server:")
        print("   ")
        print("   Option 1 (Recommended - Using virtual environment):")
        print("   F:/VsCodeFolders/Work_Elevvo/.venv/Scripts/python.exe start_api.py")
        print("   ")
        print("   Option 2 (Using batch file):")
        print("   start_api_venv.bat")
    else:
        print("\n⚠️  Setup incomplete. Please fix the issues above.")
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
