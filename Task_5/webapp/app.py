"""
Flask web application for traffic sign recognition.
Professional implementation with error handling, logging, and proper structure.
"""
import os
import sys
from pathlib import Path
from flask import Flask, request, render_template, jsonify, send_from_directory, flash, redirect, url_for
from werkzeug.utils import secure_filename
import logging

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from model_loader import TrafficSignPredictor
from utils import setup_logging, process_uploaded_image, cleanup_old_files, format_model_info, get_confidence_color


class TrafficSignApp:
    """Main application class for traffic sign recognition."""
    
    def __init__(self):
        self.app = Flask(__name__)
        self.setup_config()
        self.setup_logging()
        self.setup_model()
        self.setup_routes()
        
    def setup_config(self):
        """Setup Flask application configuration."""
        self.app.config.update(
            SECRET_KEY=os.environ.get('SECRET_KEY', 'dev-key-change-in-production'),
            MAX_CONTENT_LENGTH=10 * 1024 * 1024,  # 10MB max file size
            UPLOAD_FOLDER=Path(__file__).parent / 'static' / 'uploads',
            MODEL_PATH=Path(__file__).parent.parent / 'production' / 'model.pt',
            CONFIG_PATH=Path(__file__).parent.parent / 'data' / 'processed' / 'preprocessing_config.json'
        )
        
        # Create upload directory
        self.app.config['UPLOAD_FOLDER'].mkdir(parents=True, exist_ok=True)
    
    def setup_logging(self):
        """Setup application logging."""
        self.logger = setup_logging()
        self.logger.info("Traffic Sign Recognition App starting...")
    
    def setup_model(self):
        """Initialize the ML model."""
        try:
            model_path = self.app.config['MODEL_PATH']
            config_path = self.app.config['CONFIG_PATH']
            
            if not model_path.exists():
                raise FileNotFoundError(f"Model file not found: {model_path}")
            if not config_path.exists():
                raise FileNotFoundError(f"Config file not found: {config_path}")
            
            self.predictor = TrafficSignPredictor(
                model_path=str(model_path),
                config_path=str(config_path)
            )
            
            self.model_info = format_model_info(self.predictor.get_model_info())
            self.logger.info("Model loaded successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to load model: {str(e)}")
            raise
    
    def setup_routes(self):
        """Setup Flask routes."""
        
        @self.app.route('/')
        def index():
            """Main page."""
            return render_template('index.html', model_info=self.model_info)
        
        @self.app.route('/predict', methods=['POST'])
        def predict():
            """Handle prediction requests."""
            try:
                # Check if file was uploaded
                if 'file' not in request.files:
                    return jsonify({'error': 'No file uploaded'}), 400
                
                file = request.files['file']
                
                # Process uploaded image
                filename, image, error = process_uploaded_image(
                    file, str(self.app.config['UPLOAD_FOLDER'])
                )
                
                if error:
                    return jsonify({'error': error}), 400
                
                # Make prediction
                predictions = self.predictor.predict(image, top_k=5)
                
                # Add confidence colors for frontend
                for pred in predictions:
                    pred['confidence_color'] = get_confidence_color(pred['confidence'])
                
                self.logger.info(f"Prediction made for {filename}: {predictions[0]['class_name']} ({predictions[0]['confidence_percent']})")
                
                return jsonify({
                    'success': True,
                    'filename': filename,
                    'predictions': predictions,
                    'image_url': url_for('uploaded_file', filename=filename)
                })
                
            except Exception as e:
                self.logger.error(f"Prediction error: {str(e)}")
                return jsonify({'error': f'Prediction failed: {str(e)}'}), 500
        
        @self.app.route('/uploads/<filename>')
        def uploaded_file(filename):
            """Serve uploaded files."""
            return send_from_directory(self.app.config['UPLOAD_FOLDER'], filename)
        
        @self.app.route('/health')
        def health_check():
            """Health check endpoint."""
            try:
                # Basic model check
                model_status = 'loaded' if hasattr(self, 'predictor') else 'not_loaded'
                
                return jsonify({
                    'status': 'healthy',
                    'model_status': model_status,
                    'model_info': self.model_info
                })
            except Exception as e:
                return jsonify({
                    'status': 'unhealthy',
                    'error': str(e)
                }), 500
        
        @self.app.route('/api/model-info')
        def api_model_info():
            """API endpoint for model information."""
            return jsonify(self.model_info)
        
        @self.app.errorhandler(413)
        def too_large(e):
            """Handle file too large error."""
            return jsonify({'error': 'File too large. Maximum size is 10MB.'}), 413
        
        @self.app.errorhandler(404)
        def not_found(e):
            """Handle 404 errors."""
            return render_template('404.html'), 404
        
        @self.app.errorhandler(500)
        def internal_error(e):
            """Handle 500 errors."""
            self.logger.error(f"Internal server error: {str(e)}")
            return render_template('500.html'), 500
    
    def run(self, host='127.0.0.1', port=5000, debug=False):
        """Run the Flask application."""
        # Cleanup old files on startup
        cleanup_old_files(str(self.app.config['UPLOAD_FOLDER']))
        
        self.logger.info(f"Starting server on {host}:{port}")
        self.app.run(host=host, port=port, debug=debug)


def create_app():
    """Application factory function."""
    app_instance = TrafficSignApp()
    return app_instance.app


if __name__ == '__main__':
    # Create and run the application
    app_instance = TrafficSignApp()
    app_instance.run(debug=True)