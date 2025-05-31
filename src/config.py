"""
Configuration management module for the project.
"""
import json
import os
from typing import Dict, Any

CONFIG_FILE_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config.json")

class Config:
    """
    Configuration manager for the application.
    Loads configuration from config.json file and provides access to settings.
    """
    _instance = None
    _config = {}

    def __new__(cls):
        """Singleton pattern implementation for Config class."""
        if cls._instance is None:
            cls._instance = super(Config, cls).__new__(cls)
            cls._instance._load_config()
        return cls._instance

    def _load_config(self) -> None:
        """
        Load configuration from the config file.
        """
        try:
            with open(CONFIG_FILE_PATH, "r", encoding="utf-8") as f:
                self._config = json.load(f)
        except FileNotFoundError:
            print(f"Warning: Config file not found at {CONFIG_FILE_PATH}. Using default configuration.")
            self._config = {}

    def get(self, key_path: str, default: Any = None) -> Any:
        """
        Get a configuration value by its key path.
        
        Args:
            key_path: Dot-separated path to the configuration value (e.g., 'app.name')
            default: Default value to return if the key doesn't exist
            
        Returns:
            The configuration value or the default if not found
        """
        keys = key_path.split('.')
        value = self._config
        
        for key in keys:
            if not isinstance(value, dict) or key not in value:
                return default
            value = value[key]
            
        return value

    def set(self, key_path: str, value: Any) -> None:
        """
        Set a configuration value by its key path.
        
        Args:
            key_path: Dot-separated path to the configuration value (e.g., 'app.name')
            value: The value to set
        """
        keys = key_path.split('.')
        config = self._config
        
        for key in keys[:-1]:
            if key not in config:
                config[key] = {}
            config = config[key]
            
        config[keys[-1]] = value

    def save(self) -> None:
        """Save the current configuration to the config file."""
        with open(CONFIG_FILE_PATH, "w", encoding="utf-8") as f:
            json.dump(self._config, f, indent=4)

    @property
    def all(self) -> Dict[str, Any]:
        """Get all configuration values."""
        return self._config.copy()


# Create a singleton instance for easy importing
config = Config()