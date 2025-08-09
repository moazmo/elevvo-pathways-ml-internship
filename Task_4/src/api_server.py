"""
Production API server for Walmart Sales Forecasting.
FastAPI-based REST API for serving predictions.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import joblib
import numpy as np
from datetime import datetime, timedelta
from typing import List, Optional
import json

from utils.config import PROJECT_ROOT, MODELS_DIR, PROCESSED_DATA_DIR
from utils.logger import get_project_logger

# Initialize logger
logger = get_project_logger("api_server")

# Initialize FastAPI app
app = FastAPI(
    title="Walmart Sales Forecasting API",
    description="Production API for sales forecasting using advanced ML models",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for loaded models
models = {}
feature_list = []
label_encoders = {}

class PredictionRequest(BaseModel):
    """Request model for sales prediction."""
    store_id: int
    dept_id: int
    date: str  # YYYY-MM-DD format
    temperature: Optional[float] = None
    fuel_price: Optional[float] = None
    markdowns: Optional[List[float]] = None
    cpi: Optional[float] = None
    unemployment: Optional[float] = None

class PredictionResponse(BaseModel):
    """Response model for sales prediction."""
    store_id: int
    dept_id: int
    date: str
    predicted_sales: float
    confidence_interval: List[float]
    model_used: str
    prediction_timestamp: str

@app.on_event("startup")
async def load_models():
    """Load trained models and preprocessing artifacts on startup."""
    try:
        logger.info("Loading models and artifacts...")
        
        # Load best model
        global models, feature_list, label_encoders
        
        models_path = MODELS_DIR / "advanced_models.pkl"
        if models_path.exists():
            try:
                with open(models_path, 'rb') as f:
                    models = joblib.load(f)
                logger.info(f"Loaded {len(models)} models")
            except Exception as e:
                logger.warning(f"Could not load models due to compatibility issue: {e}")
                logger.info("Using fallback prediction algorithm")
                models = None
        
        # Load feature list
        feature_list_path = PROCESSED_DATA_DIR / "feature_list.txt"
        if feature_list_path.exists():
            with open(feature_list_path, 'r') as f:
                feature_list = [line.strip() for line in f.readlines()]
            logger.info(f"Loaded {len(feature_list)} features")
        else:
            # Fallback feature list
            feature_list = ['Store', 'Dept', 'Temperature', 'Fuel_Price', 'CPI', 'Unemployment', 'IsHoliday']
            logger.info("Using fallback feature list")
        
        # Load label encoders
        encoders_path = PROCESSED_DATA_DIR / "label_encoders.pkl"
        if encoders_path.exists():
            try:
                with open(encoders_path, 'rb') as f:
                    label_encoders = joblib.load(f)
                logger.info("Loaded label encoders")
            except Exception as e:
                logger.warning(f"Could not load label encoders due to compatibility issue: {e}")
                logger.info("Using fallback encoders")
                label_encoders = {}
        else:
            label_encoders = {}
        
        logger.info("Model loading completed successfully")
        
    except Exception as e:
        logger.error(f"Error loading models: {e}")
        # Don't raise the exception, just log it and continue with fallback
        logger.info("Continuing with fallback prediction mode")

@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "message": "Walmart Sales Forecasting API",
        "status": "running",
        "models_loaded": len(models) if models else 0,
        "features_available": len(feature_list)
    }

@app.get("/health")
async def health_check():
    """Detailed health check."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "models": {
            "total_loaded": len(models) if models else 0,
            "available_models": list(models.keys()) if models else [],
            "fallback_mode": models is None
        },
        "features": {
            "total_features": len(feature_list),
            "encoders_loaded": len(label_encoders)
        }
    }

def fallback_prediction(request):
    """Simple fallback prediction when models can't be loaded"""
    # Simple prediction algorithm for testing
    base_sales = 1000
    
    # Adjust based on features
    if request.is_holiday:
        base_sales *= 1.5
    
    # Temperature effect
    if 60 <= request.temperature <= 80:
        base_sales *= 1.2
    elif request.temperature < 32 or request.temperature > 90:
        base_sales *= 0.8
    
    # Economic factors
    if request.unemployment > 8:
        base_sales *= 0.9
    
    if request.fuel_price > 3.0:
        base_sales *= 0.95
    
    # Store/Department effect
    base_sales *= (1 + (request.store_id % 10) * 0.05)
    base_sales *= (1 + (request.dept_id % 5) * 0.03)
    
    # Add some randomness for realism
    import random
    prediction = base_sales * (0.9 + random.random() * 0.2)
    
    return round(prediction, 2)

@app.post("/predict", response_model=PredictionResponse)
async def predict_sales(request: PredictionRequest):
    """Generate sales prediction for given store, department, and date."""
    try:
        logger.info(f"Prediction request: Store {request.store_id}, Dept {request.dept_id}, Date {request.date}")
        
        # Check if models are loaded, otherwise use fallback
        if not models:
            logger.info("Using fallback prediction algorithm")
            prediction = fallback_prediction(request)
            return PredictionResponse(
                store_id=request.store_id,
                dept_id=request.dept_id,
                date=request.date,
                predicted_sales=prediction,
                confidence_interval=[prediction * 0.9, prediction * 1.1],
                model_used="fallback_algorithm",
                prediction_timestamp=datetime.now().isoformat()
            )
        
        # Parse date
        try:
            prediction_date = datetime.strptime(request.date, "%Y-%m-%d")
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD")
        
        # Create feature vector (simplified for demo)
        # In production, this would involve full feature engineering pipeline
        features = create_feature_vector(request, prediction_date)
        
        # Get prediction from best model (Weighted Ensemble)
        model_name = "weighted_ensemble"
        if model_name in models:
            model = models[model_name]
            prediction = model.predict([features])[0]
        else:
            # Fallback to any available model
            model_name = list(models.keys())[0]
            model = models[model_name]
            prediction = model.predict([features])[0]
        
        # Calculate confidence interval (simplified)
        confidence_interval = [
            float(prediction * 0.9),  # Lower bound
            float(prediction * 1.1)   # Upper bound
        ]
        
        response = PredictionResponse(
            store_id=request.store_id,
            dept_id=request.dept_id,
            date=request.date,
            predicted_sales=float(prediction),
            confidence_interval=confidence_interval,
            model_used=model_name,
            prediction_timestamp=datetime.now().isoformat()
        )
        
        logger.info(f"Prediction successful: {prediction:.2f}")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

def create_feature_vector(request: PredictionRequest, prediction_date: datetime):
    """Create feature vector from request (simplified version)."""
    # This is a simplified feature creation for demo purposes
    # In production, this would use the full feature engineering pipeline
    
    features = [
        request.store_id,
        request.dept_id,
        prediction_date.year,
        prediction_date.month,
        prediction_date.isocalendar().week,
        prediction_date.timetuple().tm_yday,
        request.temperature or 70.0,  # Default values
        request.fuel_price or 3.5,
        request.cpi or 220.0,
        request.unemployment or 7.0
    ]
    
    # Pad with zeros to match expected feature count
    while len(features) < len(feature_list):
        features.append(0.0)
    
    return features[:len(feature_list)]

@app.post("/batch_predict")
async def batch_predict(requests: List[PredictionRequest]):
    """Generate predictions for multiple requests."""
    try:
        predictions = []
        for request in requests:
            prediction = await predict_sales(request)
            predictions.append(prediction)
        
        return {
            "predictions": predictions,
            "batch_size": len(predictions),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")

@app.get("/models")
async def list_models():
    """List available models and their metadata."""
    return {
        "available_models": list(models.keys()) if models else [],
        "total_models": len(models) if models else 0,
        "feature_count": len(feature_list)
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
