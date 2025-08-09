"""
Production Startup Script for Walmart Sales Forecasting API
=========================================================

This script starts the production-ready FastAPI server for sales forecasting.

Usage:
    python start_api.py

Features:
    - Automatic model loading
    - Health monitoring endpoints
    - Professional logging
    - Error handling and recovery

API Endpoints:
    - GET  /          : Basic health check
    - GET  /health    : Detailed system status
    - POST /predict   : Single prediction
    - POST /batch_predict : Batch predictions
    - GET  /models    : Available models info

Access:
    - API Documentation: http://localhost:8000/docs
    - Health Check: http://localhost:8000/health
"""

import sys
import os
from pathlib import Path

# Add project root to Python path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

def main():
    """Start the production API server."""
    
    print("🚀 Walmart Sales Forecasting API")
    print("=" * 50)
    print("Starting production server...")
    print(f"Project Root: {PROJECT_ROOT}")
    print("=" * 50)
    
    try:
        # Import dependencies
        import uvicorn
        from src.api_server import app
        
        print("✅ Dependencies loaded successfully")
        print("📊 API Documentation: http://localhost:8000/docs")
        print("🔍 Health Check: http://localhost:8000/health")
        print("🔄 Starting server...")
        print("=" * 50)
        
        # Start the server
        uvicorn.run(
            app, 
            host="0.0.0.0", 
            port=8000,
            log_level="info",
            reload=False,  # Set to True for development
            workers=1      # Increase for production load
        )
        
    except ImportError as e:
        print(f"❌ Import Error: {e}")
        print("💡 Make sure you've installed all requirements:")
        print("   pip install -r requirements.txt")
        sys.exit(1)
        
    except Exception as e:
        print(f"❌ Server Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
