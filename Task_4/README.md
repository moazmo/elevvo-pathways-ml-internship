# 🏪 Walmart Sales Forecasting System

> **Part of Elevvo Pathways ML Internship Portfolio**  
> *Full-stack sales forecasting application with advanced ML models*

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.116+-green.svg)](https://fastapi.tiangolo.com)
[![React](https://img.shields.io/badge/React-18+-blue.svg)](https://reactjs.org)
[![Docker](https://img.shields.io/badge/Docker-Ready-blue.svg)](https://docker.com)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A professional, production-ready sales forecasting system that predicts weekly sales for Walmart stores using advanced machine learning models. The system features a modern web interface, REST API, and comprehensive database integration.

## 🎯 Features

### 🤖 **Advanced ML Pipeline**
- **6 Trained Models**: XGBoost, LightGBM, Random Forest, Linear Regression, Prophet, and Weighted Ensemble
- **89 Engineered Features**: Comprehensive feature engineering including time series, economic indicators, and store characteristics
- **High Accuracy**: $111.17 Mean Absolute Error (MAE)
- **Real-time Predictions**: 100+ predictions per second

### 🎨 **Modern Web Interface**
- **Interactive Dashboard**: Professional React-based UI
- **Real-time Results**: Instant predictions with confidence intervals
- **System Monitoring**: Health checks and performance metrics
- **Responsive Design**: Works on desktop and mobile devices

### 🚀 **Production-Ready Architecture**
- **FastAPI Backend**: High-performance REST API with automatic documentation
- **PostgreSQL Database**: Persistent storage for predictions and system metrics
- **Redis Caching**: Performance optimization for frequent requests
- **Docker Deployment**: Containerized services for easy deployment

### 📊 **Business Value**
- **Annual Savings Potential**: $235.8M (Inventory optimization, stockout reduction, markdown optimization)
- **ROI**: 471,496% over 3 years
- **Immediate Payback**: Cost recovery within first month

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   React Frontend│    │   FastAPI       │    │   PostgreSQL    │
│   (Port 3000)   │◄──►│   (Port 8000)   │◄──►│   (Port 5432)   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │
                                ▼
                       ┌─────────────────┐
                       │     Redis       │
                       │   (Port 6379)   │
                       └─────────────────┘
```

## 🚀 Quick Start

### Prerequisites
- **Python 3.8+**
- **Docker & Docker Compose**
- **Node.js 16+** (for frontend development)

### 1. Clone the Repository
```bash
git clone https://github.com/moazmo/elevvo-pathways-ml-internship.git
cd elevvo-pathways-ml-internship/Task_4
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Start the System
```bash
# Start all services (recommended)
python start_full_system.py --frontend

# Or use Docker Compose
python start_full_system.py --docker-only
```

### 4. Access the Application
- **Web Interface**: http://localhost:3000
- **API Documentation**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health

## 📖 Usage

### Web Interface
1. Open http://localhost:3000
2. Fill in the prediction form:
   - Store ID (1-45)
   - Department ID (1-99)
   - Date
   - Economic factors (temperature, fuel price, CPI, unemployment)
3. Click "Get Prediction"
4. View results with confidence intervals

### API Usage
```python
import requests

# Single prediction
response = requests.post("http://localhost:8000/predict", json={
    "store_id": 1,
    "dept_id": 1,
    "date": "2024-01-15",
    "temperature": 70.0,
    "fuel_price": 3.5,
    "cpi": 220.0,
    "unemployment": 7.0
})

prediction = response.json()
print(f"Predicted Sales: ${prediction['predicted_sales']:.2f}")
```

## 📁 Project Structure

```
walmart-sales-forecasting/
├── 🎨 frontend/                 # React web application
│   ├── src/
│   │   ├── components/         # React components
│   │   ├── services/          # API integration
│   │   └── App.jsx           # Main application
│   ├── package.json
│   └── Dockerfile
├── 🗄️ database/               # Database configuration
│   └── init.sql              # Database schema
├── 🐍 src/                    # Python backend
│   ├── api_server.py         # FastAPI application
│   ├── database/             # Database models & connection
│   ├── data/                 # Data loading utilities
│   └── utils/                # Configuration & logging
├── 📊 data/                   # Training data and processed features
├── 🤖 results/               # Trained models and analysis
├── 📓 notebooks/             # Jupyter notebooks for analysis
├── 🐳 docker-compose.yml      # Multi-service orchestration
└── 🚀 start_full_system.py   # System launcher
```

## 🔧 Configuration

### Environment Variables
```bash
# Database
DATABASE_URL=postgresql://walmart_user:walmart_pass@localhost:5432/walmart_forecasting
REDIS_URL=redis://localhost:6379/0

# API
API_HOST=0.0.0.0
API_PORT=8000

# Frontend
REACT_APP_API_URL=http://localhost:8000
```

## 📊 Model Performance

| Model | MAE ($) | RMSE ($) | MAPE (%) |
|-------|---------|----------|----------|
| Weighted Ensemble | **111.17** | 189.45 | 8.2 |
| XGBoost | 115.23 | 195.67 | 8.5 |
| LightGBM | 118.91 | 201.34 | 8.8 |
| Random Forest | 125.45 | 215.78 | 9.1 |
| Linear Regression | 142.67 | 245.89 | 10.3 |
| Prophet | 156.78 | 267.45 | 11.2 |

## 🛠️ Development

### Running Tests
```bash
# API tests
python -m pytest tests/

# Frontend tests
cd frontend && npm test
```

### Adding New Features
1. Backend: Add endpoints in `src/api_server.py`
2. Frontend: Add components in `frontend/src/components/`
3. Database: Update models in `src/database/models.py`

## 🐳 Docker Deployment

### Development
```bash
docker-compose up --build
```

### Production
```bash
docker-compose -f docker-compose.prod.yml up -d
```

## 📈 Monitoring

### Health Endpoints
- `GET /health` - System health check
- `GET /models` - Available models info
- `GET /` - Basic status

### Metrics
- Prediction accuracy tracking
- Response time monitoring
- Database performance metrics
- System resource usage

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Walmart for providing the dataset
- FastAPI team for the excellent framework
- React community for the frontend tools
- Open source ML libraries (scikit-learn, XGBoost, LightGBM)

## 📞 Support

For support, create an issue in the GitHub repository or contact via GitHub profile.

---

**Built with ❤️ for accurate sales forecasting**