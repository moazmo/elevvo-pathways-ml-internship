"""
Logging utilities for the sales forecasting project.
"""

import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional

def setup_logger(
    name: str = "sales_forecasting",
    level: int = logging.INFO,
    log_file: Optional[str] = None,
    console_output: bool = True
) -> logging.Logger:
    """
    Set up a logger with both file and console handlers.
    
    Args:
        name: Logger name
        level: Logging level
        log_file: Path to log file (optional)
        console_output: Whether to output to console
    
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    # File handler
    if log_file:
        # Ensure log directory exists
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

def get_project_logger(module_name: str) -> logging.Logger:
    """
    Get a project-specific logger for a module.
    
    Args:
        module_name: Name of the module requesting the logger
    
    Returns:
        Configured logger instance
    """
    # Create logs directory
    logs_dir = Path(__file__).parent.parent.parent / "logs"
    logs_dir.mkdir(exist_ok=True)
    
    # Create log file with timestamp
    timestamp = datetime.now().strftime("%Y%m%d")
    log_file = logs_dir / f"sales_forecasting_{timestamp}.log"
    
    return setup_logger(
        name=f"sales_forecasting.{module_name}",
        log_file=str(log_file),
        console_output=True
    )

# Example usage
if __name__ == "__main__":
    logger = get_project_logger("test")
    logger.info("Logger setup test successful")
    logger.warning("This is a warning message")
    logger.error("This is an error message")
