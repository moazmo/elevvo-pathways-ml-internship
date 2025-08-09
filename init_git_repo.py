#!/usr/bin/env python3
"""
Git Repository Initialization Script
====================================

This script helps initialize the Elevvo Pathways ML Internship portfolio
for GitHub publication with proper Git LFS setup.

Usage:
    python init_git_repo.py
"""

import subprocess
import sys
from pathlib import Path
import os

def run_command(command, cwd=None, check=True):
    """Run a shell command and return the result."""
    try:
        result = subprocess.run(
            command, 
            shell=True, 
            cwd=cwd, 
            check=check,
            capture_output=True,
            text=True
        )
        return result
    except subprocess.CalledProcessError as e:
        print(f"❌ Command failed: {command}")
        print(f"Error: {e.stderr}")
        return None

def check_git_installed():
    """Check if Git is installed."""
    result = run_command("git --version", check=False)
    if result and result.returncode == 0:
        print(f"✅ Git installed: {result.stdout.strip()}")
        return True
    else:
        print("❌ Git is not installed. Please install Git first.")
        return False

def check_git_lfs_installed():
    """Check if Git LFS is installed."""
    result = run_command("git lfs version", check=False)
    if result and result.returncode == 0:
        print(f"✅ Git LFS installed: {result.stdout.strip()}")
        return True
    else:
        print("❌ Git LFS is not installed. Please install Git LFS first.")
        print("💡 Install from: https://git-lfs.github.io/")
        return False

def initialize_git_repo():
    """Initialize Git repository."""
    print("\n🚀 Initializing Git repository...")
    
    # Check if already a git repo
    if Path(".git").exists():
        print("✅ Git repository already exists")
        return True
    
    # Initialize git repo
    result = run_command("git init")
    if result and result.returncode == 0:
        print("✅ Git repository initialized")
        return True
    else:
        print("❌ Failed to initialize Git repository")
        return False

def setup_git_lfs():
    """Set up Git LFS."""
    print("\n📦 Setting up Git LFS...")
    
    # Install Git LFS hooks
    result = run_command("git lfs install")
    if result and result.returncode == 0:
        print("✅ Git LFS hooks installed")
    else:
        print("❌ Failed to install Git LFS hooks")
        return False
    
    # Track large files
    lfs_patterns = [
        "*.csv",
        "*.pkl",
        "*.joblib", 
        "*.pt",
        "*.pth",
        "*.h5",
        "*.model",
        "*.data",
        "*.bin",
        "*.weights",
        "*.zip",
        "*.tar.gz",
        "*.jpg",
        "*.png",
        "*.mp4"
    ]
    
    for pattern in lfs_patterns:
        result = run_command(f"git lfs track '{pattern}'")
        if result and result.returncode == 0:
            print(f"✅ Tracking {pattern} with Git LFS")
        else:
            print(f"⚠️  Failed to track {pattern}")
    
    return True

def add_initial_files():
    """Add initial files to Git."""
    print("\n📁 Adding files to Git...")
    
    # Add .gitattributes first (important for LFS)
    result = run_command("git add .gitattributes")
    if result and result.returncode == 0:
        print("✅ Added .gitattributes")
    
    # Add other configuration files
    config_files = [
        ".gitignore",
        "README.md",
        "LICENSE",
        "SETUP.md",
        "CONTRIBUTING.md",
        "PROJECT_STRUCTURE.md",
        "requirements-dev.txt",
        ".github/"
    ]
    
    for file in config_files:
        if Path(file).exists():
            result = run_command(f"git add {file}")
            if result and result.returncode == 0:
                print(f"✅ Added {file}")
    
    # Add project files (this will respect .gitignore and LFS rules)
    result = run_command("git add Task_*/README.md Task_*/requirements.txt")
    if result and result.returncode == 0:
        print("✅ Added project documentation and requirements")
    
    return True

def create_initial_commit():
    """Create initial commit."""
    print("\n💾 Creating initial commit...")
    
    commit_message = """🚀 Initial commit: Elevvo Pathways ML Internship Portfolio

- 5 comprehensive ML projects spanning classical to deep learning
- Production-ready deployments with FastAPI, Flask, and React
- Complete MLOps pipeline with testing and documentation
- Professional code structure with Git LFS for large files

Projects:
- 🛍️ Customer Segmentation Analysis (K-Means, DBSCAN)
- 🌲 Forest Cover Classification (XGBoost, LightGBM)
- 💰 Loan Approval Prediction (FastAPI, MLOps)
- 🏪 Walmart Sales Forecasting (Full-stack React app)
- 🚦 Traffic Sign Recognition (PyTorch CNN, 99.49% accuracy)

Developed during internship at Elevvo Pathways
"""
    
    result = run_command(f'git commit -m "{commit_message}"')
    if result and result.returncode == 0:
        print("✅ Initial commit created")
        return True
    else:
        print("❌ Failed to create initial commit")
        return False

def setup_remote_origin():
    """Set up remote origin (user will need to provide URL)."""
    print("\n🌐 Setting up remote origin...")
    print("📝 You'll need to:")
    print("   1. Create a new repository on GitHub")
    print("   2. Copy the repository URL")
    print("   3. Run: git remote add origin <your-repo-url>")
    print("   4. Run: git push -u origin main")
    print("\n💡 Example:")
    print("   git remote add origin https://github.com/moazmo/elevvo-pathways-ml-internship.git")
    print("   git push -u origin main")

def main():
    """Main function to initialize the repository."""
    print("🏢 Elevvo Pathways ML Internship Portfolio")
    print("🚀 Git Repository Initialization Script")
    print("=" * 60)
    
    # Check prerequisites
    if not check_git_installed():
        sys.exit(1)
    
    if not check_git_lfs_installed():
        sys.exit(1)
    
    # Initialize repository
    if not initialize_git_repo():
        sys.exit(1)
    
    # Set up Git LFS
    if not setup_git_lfs():
        sys.exit(1)
    
    # Add files
    if not add_initial_files():
        sys.exit(1)
    
    # Create initial commit
    if not create_initial_commit():
        sys.exit(1)
    
    # Instructions for remote setup
    setup_remote_origin()
    
    print("\n" + "=" * 60)
    print("🎉 Repository initialization completed!")
    print("=" * 60)
    print("\n📋 Next steps:")
    print("1. Create a new repository on GitHub")
    print("2. Add remote origin and push:")
    print("   git remote add origin <your-repo-url>")
    print("   git branch -M main")
    print("   git push -u origin main")
    print("\n💡 The repository is now ready for professional presentation!")
    print("🔗 All large files are properly configured with Git LFS")
    print("📚 Comprehensive documentation is included")
    print("🧪 CI/CD pipeline is configured")

if __name__ == "__main__":
    main()