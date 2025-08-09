"""
FastAPI server for loan approval prediction service.
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import uvicorn
from typing import List, Dict, Any
from pathlib import Path
import json
from datetime import datetime
from loguru import logger

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.production_pipeline import (
    ProductionPipeline, LoanApplicationRequest, LoanPredictionResponse,
    ModelRegistry
)
from src.config_loader import config


# Initialize FastAPI app
app = FastAPI(
    title="Loan Approval Prediction API",
    description="Professional loan approval prediction service using machine learning",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Setup static files and templates
webapp_dir = Path(__file__).parent.parent / "webapp"
app.mount("/static", StaticFiles(directory=str(webapp_dir / "static")), name="static")
templates = Jinja2Templates(directory=str(webapp_dir / "templates"))

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global pipeline instance
pipeline: ProductionPipeline = None
model_registry: ModelRegistry = None


@app.on_event("startup")
async def startup_event():
    """Initialize the prediction pipeline on startup."""
    global pipeline, model_registry
    
    try:
        # Initialize model registry
        model_registry = ModelRegistry()
        
        # Get active model info
        active_model_info = model_registry.get_active_model_info()
        
        if active_model_info:
            model_path = active_model_info["model_path"]
            pipeline = ProductionPipeline(model_path=model_path)
            logger.info(f"Production pipeline initialized with model: {active_model_info['version_id']}")
        else:
            logger.warning("No active model found in registry")
            
    except Exception as e:
        logger.error(f"Failed to initialize pipeline: {e}")


@app.get("/", response_class=HTMLResponse)
async def web_app(request: Request):
    """Serve the main web application."""
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/admin", response_class=HTMLResponse)
async def admin_dashboard(request: Request):
    """Serve the admin dashboard."""
    return templates.TemplateResponse("admin.html", {"request": request})

@app.get("/api")
async def api_root():
    """API root endpoint with information."""
    return {
        "message": "Loan Approval Prediction API",
        "version": "1.0.0",
        "status": "active",
        "endpoints": {
            "predict": "/predict",
            "predict_batch": "/predict/batch",
            "health": "/health",
            "model_info": "/model/info",
            "docs": "/docs"
        }
    }


@app.post("/predict", response_model=LoanPredictionResponse)
async def predict_loan_approval(application: LoanApplicationRequest):
    """
    Predict loan approval for a single application.
    
    Args:
        application: Loan application data
        
    Returns:
        Loan prediction response
    """
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Prediction service not available")
    
    try:
        # Convert Pydantic model to dict
        application_data = application.dict()
        
        # Make prediction
        prediction_response = pipeline.predict(application_data)
        
        logger.info(f"Prediction made for loan_id: {application.loan_id}")
        
        return prediction_response
        
    except ValueError as e:
        logger.error(f"Validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.post("/predict/batch", response_model=List[LoanPredictionResponse])
async def predict_batch_loan_approval(applications: List[LoanApplicationRequest]):
    """
    Predict loan approval for multiple applications.
    
    Args:
        applications: List of loan application data
        
    Returns:
        List of loan prediction responses
    """
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Prediction service not available")
    
    if len(applications) > 100:  # Limit batch size
        raise HTTPException(status_code=400, detail="Batch size too large (max 100)")
    
    try:
        # Convert Pydantic models to dicts
        applications_data = [app.dict() for app in applications]
        
        # Make batch predictions
        prediction_responses = pipeline.predict_batch(applications_data)
        
        logger.info(f"Batch prediction completed for {len(applications)} applications")
        
        return prediction_responses
        
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    if pipeline is None:
        return JSONResponse(
            status_code=503,
            content={"status": "unhealthy", "message": "Pipeline not initialized"}
        )
    
    try:
        health_status = pipeline.health_check()
        
        if health_status["status"] == "healthy":
            return JSONResponse(status_code=200, content=health_status)
        else:
            return JSONResponse(status_code=503, content=health_status)
            
    except Exception as e:
        logger.error(f"Health check error: {e}")
        return JSONResponse(
            status_code=503,
            content={"status": "unhealthy", "message": str(e)}
        )


@app.get("/model/info")
async def get_model_info():
    """Get information about the currently loaded model."""
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    
    try:
        model_info = pipeline.get_model_info()
        
        # Add registry information
        if model_registry:
            active_model_info = model_registry.get_active_model_info()
            if active_model_info:
                model_info["registry_info"] = active_model_info
        
        return model_info
        
    except Exception as e:
        logger.error(f"Error getting model info: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.get("/models/list")
async def list_models():
    """List all registered models."""
    if model_registry is None:
        raise HTTPException(status_code=503, detail="Model registry not available")
    
    try:
        models = model_registry.list_models()
        return {"models": models}
        
    except Exception as e:
        logger.error(f"Error listing models: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


def run_server():
    """Run the FastAPI server."""
    production_config = config.get_production_config()
    
    uvicorn.run(
        "src.api_server:app",
        host=production_config['api_host'],
        port=production_config['api_port'],
        log_level=production_config['log_level'].lower(),
        reload=False  # Set to True for development
    )


if __name__ == "__main__":
    run_server()
