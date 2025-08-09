"""
Database models for the Walmart Sales Forecasting application.
"""

from sqlalchemy import Column, Integer, String, DateTime, Boolean, Text, JSON
from sqlalchemy.types import DECIMAL
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.sql import func
from datetime import datetime

Base = declarative_base()

class Prediction(Base):
    """Model for storing sales predictions."""
    __tablename__ = 'predictions'
    
    id = Column(Integer, primary_key=True, index=True)
    store_id = Column(Integer, nullable=False, index=True)
    dept_id = Column(Integer, nullable=False, index=True)
    prediction_date = Column(DateTime, nullable=False, index=True)
    predicted_sales = Column(DECIMAL(12, 2), nullable=False)
    confidence_lower = Column(DECIMAL(12, 2))
    confidence_upper = Column(DECIMAL(12, 2))
    model_used = Column(String(100))
    input_features = Column(JSON)
    created_at = Column(DateTime, default=func.now(), index=True)

class UserSession(Base):
    """Model for user sessions."""
    __tablename__ = 'user_sessions'
    
    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String(255), unique=True, nullable=False, index=True)
    user_data = Column(JSON)
    created_at = Column(DateTime, default=func.now())
    expires_at = Column(DateTime)
    is_active = Column(Boolean, default=True)

class SystemMetric(Base):
    """Model for system metrics."""
    __tablename__ = 'system_metrics'
    
    id = Column(Integer, primary_key=True, index=True)
    metric_name = Column(String(100), nullable=False, index=True)
    metric_value = Column(DECIMAL(12, 4))
    metric_data = Column(JSON)
    recorded_at = Column(DateTime, default=func.now(), index=True)

class BatchJob(Base):
    """Model for batch processing jobs."""
    __tablename__ = 'batch_jobs'
    
    id = Column(Integer, primary_key=True, index=True)
    job_id = Column(String(255), unique=True, nullable=False, index=True)
    job_type = Column(String(50), nullable=False)
    status = Column(String(20), default='pending', index=True)
    input_data = Column(JSON)
    results = Column(JSON)
    error_message = Column(Text)
    created_at = Column(DateTime, default=func.now())
    started_at = Column(DateTime)
    completed_at = Column(DateTime)