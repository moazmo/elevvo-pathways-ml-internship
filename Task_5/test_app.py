"""
Test script for the traffic sign recognition web application.
"""
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

def test_model_loading():
    """Test if the model can be loaded successfully."""
    try:
        from model_loader import TrafficSignPredictor
        
        model_path = Path(__file__).parent / 'production' / 'model.pt'
        config_path = Path(__file__).parent / 'data' / 'processed' / 'preprocessing_config.json'
        
        print("Testing model loading...")
        print(f"Model path: {model_path}")
        print(f"Config path: {config_path}")
        
        if not model_path.exists():
            print("❌ Model file not found!")
            return False
            
        if not config_path.exists():
            print("❌ Config file not found!")
            return False
        
        predictor = TrafficSignPredictor(
            model_path=str(model_path),
            config_path=str(config_path)
        )
        
        model_info = predictor.get_model_info()
        print("✅ Model loaded successfully!")
        print(f"Model info: {model_info}")
        
        return True
        
    except Exception as e:
        print(f"❌ Model loading failed: {str(e)}")
        return False

def test_flask_app():
    """Test if the Flask app can be created."""
    try:
        sys.path.append(str(Path(__file__).parent / 'webapp'))
        from app import create_app
        
        print("\nTesting Flask app creation...")
        app = create_app()
        
        with app.test_client() as client:
            # Test health endpoint
            response = client.get('/health')
            print(f"Health check status: {response.status_code}")
            
            # Test main page
            response = client.get('/')
            print(f"Main page status: {response.status_code}")
        
        print("✅ Flask app created successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Flask app creation failed: {str(e)}")
        return False

if __name__ == '__main__':
    print("🚀 Testing Traffic Sign Recognition App\n")
    
    model_ok = test_model_loading()
    app_ok = test_flask_app()
    
    print(f"\n📊 Test Results:")
    print(f"Model Loading: {'✅ PASS' if model_ok else '❌ FAIL'}")
    print(f"Flask App: {'✅ PASS' if app_ok else '❌ FAIL'}")
    
    if model_ok and app_ok:
        print("\n🎉 All tests passed! Ready to run the application.")
        print("\nTo start the app, run:")
        print("cd webapp && python app.py")
    else:
        print("\n⚠️  Some tests failed. Please check the errors above.")