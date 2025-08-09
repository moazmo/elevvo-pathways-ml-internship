# 🚦 Traffic Sign Recognition

> **Part of Elevvo Pathways ML Internship Portfolio**  
> *Deep learning web application for traffic sign classification*

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/DL-PyTorch-red.svg)](https://pytorch.org)
[![Flask](https://img.shields.io/badge/Web-Flask-green.svg)](https://flask.palletsprojects.com)
[![Accuracy](https://img.shields.io/badge/Accuracy-99.49%25-brightgreen.svg)](https://github.com)

A professional deep learning web application for German traffic sign classification using PyTorch and Flask.

## 🌟 Features

- **High Accuracy**: 99.49% validation accuracy with custom CNN
- **Real-time Prediction**: Fast inference with confidence scores
- **Professional UI**: Modern, responsive web interface
- **Drag & Drop**: Easy image upload with preview
- **Top-K Predictions**: Shows multiple predictions with confidence levels
- **Production Ready**: Robust error handling and logging

## 🏗️ Architecture

### Machine Learning Pipeline
- **Dataset**: German Traffic Sign Recognition Benchmark (GTSRB)
- **Model**: Custom CNN with 4 convolutional blocks
- **Framework**: PyTorch with torchvision
- **Preprocessing**: Data augmentation, normalization, stratified splitting
- **Evaluation**: Comprehensive metrics and confusion matrix analysis

### Web Application
- **Backend**: Flask with professional structure
- **Frontend**: Bootstrap 5 with modern UI/UX
- **API**: RESTful endpoints for predictions and health checks
- **Security**: Input validation, file size limits, secure uploads

## 📊 Model Performance

| Model | Validation Accuracy | Parameters | Inference Speed |
|-------|-------------------|------------|----------------|
| Custom CNN | **99.49%** ⭐ | 1.6M | 7.4ms |
| MobileNet v2 | 1.5% | 2.3M | 8.7ms |

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- PyTorch 2.0+
- 4GB+ RAM recommended

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/moazmo/elevvo-pathways-ml-internship.git
   cd elevvo-pathways-ml-internship/Task_5
   ```

2. **Create virtual environment**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download dataset**
   - Download GTSRB dataset from [Kaggle](https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign)
   - Extract to `data/raw/` directory

5. **Run training notebooks** (Optional - for training from scratch)
   ```bash
   jupyter notebook notebooks/
   ```
   Execute notebooks in order: 01 → 02 → 03 → 04

6. **Test the application**
   ```bash
   python test_app.py
   ```

7. **Start the web server**
   ```bash
   cd webapp
   python app.py
   ```

8. **Open your browser**
   Navigate to `http://127.0.0.1:5000`

## 📁 Project Structure

```
traffic-sign-recognition/
├── 📊 notebooks/           # Jupyter notebooks for ML pipeline
│   ├── 01_data_exploration.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_model_training.ipynb
│   └── 04_model_comparison.ipynb
├── 🧠 src/                 # Source code modules
│   ├── model_loader.py     # Model loading and inference
│   └── utils.py           # Utility functions
├── 🌐 webapp/             # Flask web application
│   ├── app.py             # Main Flask app
│   ├── templates/         # HTML templates
│   └── static/           # CSS, JS, uploads
├── 📁 data/              # Dataset (download required)
├── 🎯 models/            # Trained models (generated)
├── 🚀 production/        # Production model (generated)
├── 🖼️ images/            # Screenshots and documentation
└── 📋 requirements.txt   # Python dependencies
```

## 🔧 API Endpoints

- `GET /` - Main web interface
- `POST /predict` - Image prediction endpoint
- `GET /health` - Health check
- `GET /api/model-info` - Model information

## 🎯 Usage Examples

### Web Interface
1. Open the web application
2. Drag & drop or click to upload a traffic sign image
3. View real-time predictions with confidence scores

### Programmatic Usage
```python
from src.model_loader import TrafficSignPredictor
from PIL import Image

# Initialize predictor
predictor = TrafficSignPredictor(
    model_path='production/model.pt',
    config_path='data/processed/preprocessing_config.json'
)

# Load and predict
image = Image.open('path/to/traffic_sign.jpg')
predictions = predictor.predict(image, top_k=5)

for pred in predictions:
    print(f"{pred['class_name']}: {pred['confidence_percent']}")
```

## 🧪 Testing

Run the test suite to verify everything works:

```bash
python test_app.py
```

## 📈 Model Details

### Custom CNN Architecture
- **Input**: 64×64 RGB images
- **Blocks**: 4 convolutional blocks with batch normalization
- **Features**: 32 → 64 → 128 → 256 channels
- **Classifier**: 2-layer MLP with dropout
- **Activation**: ReLU with MaxPool2d

### Training Configuration
- **Optimizer**: Adam with learning rate scheduling
- **Loss**: CrossEntropyLoss
- **Augmentation**: Rotation, scaling, color jittering
- **Regularization**: Dropout, batch normalization
- **Validation**: Stratified split (80/20)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Dataset**: German Traffic Sign Recognition Benchmark (GTSRB)
- **Framework**: PyTorch team for the excellent deep learning framework
- **UI**: Bootstrap team for the responsive CSS framework

## 📞 Contact

**Moaz Mohamed** - [@moazmo](https://github.com/moazmo)

Project Link: [https://github.com/moazmo/elevvo-pathways-ml-internship](https://github.com/moazmo/elevvo-pathways-ml-internship)

---

⭐ **Star this repository if you found it helpful!**