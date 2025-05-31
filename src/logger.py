"""
Logging configuration for the project.
"""
import logging
import os
from logging.handlers import RotatingFileHandler
from typing import Optional

from .config import config

class Logger:
    """
    Logger configuration for the application.
    """
    _instance = None
    _loggers = {}

    def __new__(cls):
        """Singleton pattern implementation for Logger class."""
        if cls._instance is None:
            cls._instance = super(Logger, cls).__new__(cls)
            cls._instance._setup_logging()
        return cls._instance

    def _setup_logging(self) -> None:
        """Set up logging configuration from the config file."""
        log_level = config.get('logging.level', 'INFO')
        log_format = config.get('logging.format', 
                              '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        log_file = config.get('logging.file', 'logs/app.log')
        
        # Create logs directory if it doesn't exist
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        
        # Configure the root logger
        logging.basicConfig(
            level=getattr(logging, log_level),
            format=log_format,
            handlers=[
                RotatingFileHandler(
                    log_file, 
                    maxBytes=10485760,  # 10MB
                    backupCount=5
                ),
                logging.StreamHandler()
            ]
        )

    def get_logger(self, name: str) -> logging.Logger:
        """
        Get a logger with the specified name.
        
        Args:
            name: The name of the logger
            
        Returns:
            A configured logger instance
        """
        if name not in self._loggers:
            self._loggers[name] = logging.getLogger(name)
        return self._loggers[name]


# Create a singleton instance for easy importing
logger_factory = Logger()

def get_logger(name: Optional[str] = None) -> logging.Logger:
    """
    Get a logger for the specified module.
    
    Args:
        name: The name of the module (defaults to the caller's module name)
        
    Returns:
        A configured logger instance
    """
    if name is None:
        # Get the caller's module name if not provided
        import inspect
        frame = inspect.stack()[1]
        name = inspect.getmodule(frame[0]).__name__
    
    return logger_factory.get_logger(name)