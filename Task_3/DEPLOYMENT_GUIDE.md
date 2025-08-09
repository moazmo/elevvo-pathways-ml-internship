# 🚀 Loan Approval Web Application - Deployment Guide

## 📋 Overview

This guide provides complete instructions for deploying the Loan Approval Prediction Web Application locally. The system includes a modern web interface, REST API, and admin dashboard.

## 🏗️ System Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Web Frontend  │    │   FastAPI       │    │   ML Pipeline   │
│   (HTML/CSS/JS) │◄──►│   Backend       │◄──►│   (scikit-learn)│
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Bootstrap UI  │    │   REST API      │    │   Trained Model │
│   Chart.js      │    │   Static Files  │    │   Preprocessor  │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 🔧 Prerequisites

### System Requirements
- **Python**: 3.8 or higher
- **RAM**: Minimum 4GB (8GB recommended)
- **Storage**: 500MB free space
- **Network**: Internet connection for initial setup

### Required Python Packages
```bash
pip install fastapi uvicorn jinja2 python-multipart
pip install pandas scikit-learn joblib loguru
pip install requests  # For testing
```

## 📦 Installation Steps

### 1. Verify Model Files
Ensure these files exist:
```
models/
├── decision_tree_model.joblib
├── preprocessor.joblib
└── model_registry.json

config/
└── config.yaml
```

If missing, run:
```bash
python main_pipeline.py
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Verify Installation
```bash
python -c "from src.production_pipeline import ProductionPipeline; print('✅ Installation verified')"
```

## 🚀 Deployment Options

### Option 1: Quick Start (Recommended)
```bash
python start_webapp.py
```

### Option 2: Manual Start
```bash
python src/api_server.py
```

### Option 3: Production Mode
```bash
uvicorn src.api_server:app --host 0.0.0.0 --port 8000 --workers 1
```

## 🌐 Application URLs

| Service | URL | Description |
|---------|-----|-------------|
| **Web App** | http://localhost:8000 | Main loan application interface |
| **Admin Dashboard** | http://localhost:8000/admin | System monitoring and analytics |
| **API Documentation** | http://localhost:8000/docs | Interactive API documentation |
| **Health Check** | http://localhost:8000/health | System health status |

## 🎯 Features

### 🏠 Main Web Application
- **Responsive Design**: Works on desktop, tablet, and mobile
- **Real-time Validation**: Instant form validation with helpful messages
- **Interactive Results**: Animated prediction results with confidence scores
- **Professional UI**: Modern Bootstrap-based interface

### 📊 Admin Dashboard
- **System Monitoring**: Real-time health and performance metrics
- **Model Analytics**: Performance comparison charts and statistics
- **Activity Tracking**: Recent prediction history and usage patterns
- **Visual Analytics**: Interactive charts and graphs

### 🔌 REST API
- **Single Predictions**: `/predict` endpoint for individual applications
- **Batch Processing**: `/predict/batch` for multiple applications
- **System Health**: `/health` for monitoring
- **Model Information**: `/model/info` for current model details

## 🧪 Testing the Deployment

### 1. Basic Functionality Test
```bash
# Test API health
curl http://localhost:8000/health

# Test single prediction
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{
       "loan_id": "TEST_001",
       "no_of_dependents": 2,
       "education": "Graduate",
       "self_employed": "No",
       "income_annum": 6000000,
       "loan_amount": 15000000,
       "loan_term": 15,
       "cibil_score": 750,
       "residential_assets_value": 2500000,
       "commercial_assets_value": 1200000,
       "luxury_assets_value": 600000,
       "bank_asset_value": 400000
     }'
```

### 2. Web Interface Test
1. Open http://localhost:8000 in your browser
2. Fill out the loan application form
3. Submit and verify prediction results
4. Test the admin dashboard at http://localhost:8000/admin

### 3. Batch Processing Test
1. Navigate to the "Batch Processing" tab
2. Download the CSV template
3. Upload a test CSV file
4. Verify batch prediction results

## 🔒 Security Considerations

### For Production Deployment
- **HTTPS**: Use SSL/TLS certificates
- **Authentication**: Implement user authentication
- **Rate Limiting**: Add API rate limiting
- **Input Validation**: Enhanced input sanitization
- **Logging**: Comprehensive audit logging

### Current Security Features
- ✅ Input validation and sanitization
- ✅ Error handling and logging
- ✅ CORS configuration
- ✅ Request/response validation

## 📈 Performance Optimization

### Current Configuration
- **Single Worker**: Optimized for local deployment
- **Memory Usage**: ~200MB typical usage
- **Response Time**: <1 second for predictions
- **Throughput**: 100+ predictions/minute

### Scaling Options
- **Multiple Workers**: Increase `workers` parameter
- **Load Balancing**: Use nginx or similar
- **Caching**: Implement Redis for frequent predictions
- **Database**: Add PostgreSQL for prediction history

## 🛠️ Troubleshooting

### Common Issues

#### 1. Port Already in Use
```bash
# Find process using port 8000
netstat -ano | findstr :8000
# Kill the process (replace PID)
taskkill /PID <PID> /F
```

#### 2. Module Import Errors
```bash
# Ensure you're in the correct directory
cd Task_3
# Verify Python path
python -c "import sys; print(sys.path)"
```

#### 3. Model Loading Errors
```bash
# Regenerate model files
python main_pipeline.py
```

#### 4. Template Not Found
```bash
# Verify webapp directory structure
ls -la webapp/templates/
ls -la webapp/static/
```

## 📞 Support

### Log Files
- **Application Logs**: `logs/` directory
- **Server Logs**: Console output
- **Error Logs**: Check terminal for detailed error messages

### Health Monitoring
- **Health Endpoint**: http://localhost:8000/health
- **Model Status**: http://localhost:8000/model/info
- **System Metrics**: Available in admin dashboard

## 🎉 Success Indicators

When deployment is successful, you should see:
- ✅ Server starts without errors
- ✅ Web application loads at http://localhost:8000
- ✅ Health check returns "healthy" status
- ✅ Predictions work correctly
- ✅ Admin dashboard displays metrics

## 📝 Next Steps

1. **Test thoroughly** with various loan applications
2. **Monitor performance** using the admin dashboard
3. **Review logs** for any issues or optimization opportunities
4. **Scale as needed** based on usage patterns

---

**🎊 Congratulations! Your Loan Approval Web Application is ready for production use!**
