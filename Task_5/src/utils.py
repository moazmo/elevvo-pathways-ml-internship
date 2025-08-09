"""
Utility functions for the traffic sign recognition web application.
"""
import os
import uuid
from pathlib import Path
from typing import Optional, Tuple
from PIL import Image
import logging


def setup_logging(log_level: str = "INFO") -> logging.Logger:
    """Setup application logging."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('app.log')
        ]
    )
    return logging.getLogger(__name__)


def validate_image(file) -> Tuple[bool, Optional[str]]:
    """
    Validate uploaded image file.
    
    Args:
        file: Flask file object
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not file:
        return False, "No file provided"
    
    if file.filename == '':
        return False, "No file selected"
    
    # Check file extension
    allowed_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
    file_ext = Path(file.filename).suffix.lower()
    
    if file_ext not in allowed_extensions:
        return False, f"Invalid file type. Allowed: {', '.join(allowed_extensions)}"
    
    # Check file size (max 10MB)
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)  # Reset file pointer
    
    max_size = 10 * 1024 * 1024  # 10MB
    if file_size > max_size:
        return False, f"File too large. Maximum size: {max_size // (1024*1024)}MB"
    
    # Try to open as image
    try:
        image = Image.open(file)
        image.verify()  # Verify it's a valid image
        file.seek(0)  # Reset file pointer after verify
        return True, None
    except Exception as e:
        return False, f"Invalid image file: {str(e)}"


def process_uploaded_image(file, upload_folder: str) -> Tuple[Optional[str], Optional[Image.Image], Optional[str]]:
    """
    Process uploaded image file.
    
    Args:
        file: Flask file object
        upload_folder: Directory to save uploaded files
        
    Returns:
        Tuple of (filename, PIL_image, error_message)
    """
    # Validate image
    is_valid, error = validate_image(file)
    if not is_valid:
        return None, None, error
    
    try:
        # Generate unique filename
        file_ext = Path(file.filename).suffix.lower()
        unique_filename = f"{uuid.uuid4().hex}{file_ext}"
        
        # Create upload directory if it doesn't exist
        upload_path = Path(upload_folder)
        upload_path.mkdir(parents=True, exist_ok=True)
        
        # Save file
        file_path = upload_path / unique_filename
        file.save(str(file_path))
        
        # Load as PIL image
        image = Image.open(file_path)
        
        return unique_filename, image, None
        
    except Exception as e:
        return None, None, f"Failed to process image: {str(e)}"


def cleanup_old_files(directory: str, max_age_hours: int = 24):
    """
    Clean up old uploaded files.
    
    Args:
        directory: Directory to clean
        max_age_hours: Maximum age of files to keep
    """
    try:
        import time
        current_time = time.time()
        max_age_seconds = max_age_hours * 3600
        
        directory_path = Path(directory)
        if not directory_path.exists():
            return
        
        for file_path in directory_path.iterdir():
            if file_path.is_file():
                file_age = current_time - file_path.stat().st_mtime
                if file_age > max_age_seconds:
                    file_path.unlink()
                    
    except Exception as e:
        logging.getLogger(__name__).warning(f"Failed to cleanup files: {str(e)}")


def format_model_info(model_info: dict) -> dict:
    """Format model information for display."""
    return {
        'Model Type': model_info.get('model_type', 'Unknown'),
        'Classes': model_info.get('num_classes', 'Unknown'),
        'Input Size': f"{model_info.get('input_size', 'Unknown')}x{model_info.get('input_size', 'Unknown')}",
        'Device': model_info.get('device', 'Unknown')
    }


def get_confidence_color(confidence: float) -> str:
    """Get color class based on confidence level."""
    if confidence >= 0.8:
        return 'success'  # Green
    elif confidence >= 0.6:
        return 'warning'  # Yellow
    else:
        return 'danger'   # Red