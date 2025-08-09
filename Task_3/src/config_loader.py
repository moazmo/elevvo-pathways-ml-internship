"""
Configuration loader for the loan approval prediction system.
"""

import yaml
from pathlib import Path
from typing import Dict, Any
from loguru import logger


class ConfigLoader:
    """Handles loading and validation of configuration files."""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """
        Initialize the configuration loader.
        
        Args:
            config_path: Path to the configuration YAML file
        """
        self.config_path = Path(config_path)
        self.config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        try:
            with open(self.config_path, 'r') as file:
                config = yaml.safe_load(file)
            logger.info(f"Configuration loaded successfully from {self.config_path}")
            return config
        except FileNotFoundError:
            logger.error(f"Configuration file not found: {self.config_path}")
            raise
        except yaml.YAMLError as e:
            logger.error(f"Error parsing YAML configuration: {e}")
            raise
    
    def get(self, key_path: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation.
        
        Args:
            key_path: Dot-separated path to the configuration value (e.g., 'data.test_size')
            default: Default value if key is not found
            
        Returns:
            Configuration value or default
        """
        keys = key_path.split('.')
        value = self.config
        
        try:
            for key in keys:
                value = value[key]
            return value
        except (KeyError, TypeError):
            if default is not None:
                return default
            raise KeyError(f"Configuration key '{key_path}' not found")
    
    def get_data_config(self) -> Dict[str, Any]:
        """Get data-related configuration."""
        return self.get('data')
    
    def get_model_config(self, model_name: str) -> Dict[str, Any]:
        """Get model-specific configuration."""
        return self.get(f'models.{model_name}')
    
    def get_preprocessing_config(self) -> Dict[str, Any]:
        """Get preprocessing configuration."""
        return self.get('preprocessing')
    
    def get_evaluation_config(self) -> Dict[str, Any]:
        """Get evaluation configuration."""
        return self.get('evaluation')
    
    def get_production_config(self) -> Dict[str, Any]:
        """Get production configuration."""
        return self.get('production')


# Global configuration instance
config = ConfigLoader()
