"""
Model loading and inference utilities for traffic sign recognition.
"""
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
from PIL import Image
import torchvision.transforms as transforms


class CustomCNN(nn.Module):
    """Custom CNN architecture for traffic sign classification - matches the trained model."""
    
    def __init__(self, num_classes: int = 43):
        super(CustomCNN, self).__init__()
        
        # Feature extraction layers - matching the saved model structure
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 32, kernel_size=3, padding=1),      # features.0
            nn.BatchNorm2d(32),                              # features.1
            nn.ReLU(inplace=True),                           # features.2
            nn.Conv2d(32, 32, kernel_size=3, padding=1),     # features.3
            nn.BatchNorm2d(32),                              # features.4
            nn.ReLU(inplace=True),                           # features.5
            nn.MaxPool2d(2, 2),                              # features.6
            
            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),     # features.7
            nn.BatchNorm2d(64),                              # features.8
            nn.ReLU(inplace=True),                           # features.9
            nn.Conv2d(64, 64, kernel_size=3, padding=1),     # features.10
            nn.BatchNorm2d(64),                              # features.11
            nn.ReLU(inplace=True),                           # features.12
            nn.MaxPool2d(2, 2),                              # features.13
            
            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),    # features.14
            nn.BatchNorm2d(128),                             # features.15
            nn.ReLU(inplace=True),                           # features.16
            nn.Conv2d(128, 128, kernel_size=3, padding=1),   # features.17
            nn.BatchNorm2d(128),                             # features.18
            nn.ReLU(inplace=True),                           # features.19
            nn.MaxPool2d(2, 2),                              # features.20
            
            # Block 4
            nn.Conv2d(128, 256, kernel_size=3, padding=1),   # features.21
            nn.BatchNorm2d(256),                             # features.22
            nn.ReLU(inplace=True),                           # features.23
            nn.MaxPool2d(2, 2),                              # features.24
        )
        
        # Classifier - matching the saved model structure
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),                                 # classifier.0
            nn.Linear(256 * 4 * 4, 256),                     # classifier.1 (4096 -> 256)
            nn.ReLU(inplace=True),                           # classifier.2
            nn.Dropout(0.5),                                 # classifier.3
            nn.Linear(256, num_classes)                      # classifier.4
        )
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class TrafficSignPredictor:
    """Main predictor class for traffic sign recognition."""
    
    def __init__(self, model_path: str, config_path: str):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_path = Path(model_path)
        self.config_path = Path(config_path)
        
        # Load configuration
        self.config = self._load_config()
        
        # Initialize model
        self.model = self._load_model()
        
        # Setup preprocessing
        self.transform = self._setup_transforms()
        
        # Load class names
        self.class_names = self._load_class_names()
    
    def _load_config(self) -> Dict:
        """Load preprocessing configuration."""
        with open(self.config_path, 'r') as f:
            return json.load(f)
    
    def _load_model(self) -> nn.Module:
        """Load the trained model."""
        model = CustomCNN(num_classes=len(self.config['classes']))
        
        # Load checkpoint
        checkpoint = torch.load(self.model_path, map_location=self.device)
        
        # Handle different checkpoint formats
        if isinstance(checkpoint, dict) and 'model_state' in checkpoint:
            # New format with metadata
            model.load_state_dict(checkpoint['model_state'])
        elif isinstance(checkpoint, dict):
            # Direct state dict
            model.load_state_dict(checkpoint)
        else:
            # Fallback
            model.load_state_dict(checkpoint)
        
        model.to(self.device)
        model.eval()
        
        return model
    
    def _setup_transforms(self) -> transforms.Compose:
        """Setup image preprocessing transforms."""
        img_size = self.config['img_size']
        mean = self.config['dataset_mean_std']['mean']
        std = self.config['dataset_mean_std']['std']
        
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
    
    def _load_class_names(self) -> Dict[int, str]:
        """Load traffic sign class names."""
        # Basic class mapping - can be enhanced with actual sign names
        class_names = {}
        for i, class_id in enumerate(self.config['classes']):
            class_names[i] = f"Traffic Sign {class_id}"
        
        return class_names
    
    def preprocess_image(self, image: Image.Image) -> torch.Tensor:
        """Preprocess a PIL image for model inference."""
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Apply transforms
        tensor = self.transform(image)
        
        # Add batch dimension
        tensor = tensor.unsqueeze(0)
        
        return tensor.to(self.device)
    
    def predict(self, image: Image.Image, top_k: int = 5) -> List[Dict]:
        """
        Predict traffic sign class for an image.
        
        Args:
            image: PIL Image
            top_k: Number of top predictions to return
            
        Returns:
            List of predictions with class, confidence, and class_name
        """
        try:
            # Preprocess image
            input_tensor = self.preprocess_image(image)
            
            # Model inference
            with torch.no_grad():
                outputs = self.model(input_tensor)
                probabilities = F.softmax(outputs, dim=1)
                
                # Get top-k predictions
                top_probs, top_indices = torch.topk(probabilities, top_k)
                
                # Format results
                predictions = []
                for i in range(top_k):
                    class_idx = top_indices[0][i].item()
                    confidence = top_probs[0][i].item()
                    class_id = self.config['classes'][class_idx]
                    
                    predictions.append({
                        'class_id': class_id,
                        'class_name': self.class_names.get(class_idx, f"Class {class_id}"),
                        'confidence': float(confidence),
                        'confidence_percent': f"{confidence * 100:.2f}%"
                    })
                
                return predictions
                
        except Exception as e:
            raise RuntimeError(f"Prediction failed: {str(e)}")
    
    def get_model_info(self) -> Dict:
        """Get model information."""
        return {
            'model_type': 'Custom CNN',
            'num_classes': len(self.config['classes']),
            'input_size': self.config['img_size'],
            'device': str(self.device),
            'model_path': str(self.model_path)
        }