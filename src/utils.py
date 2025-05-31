"""
Utility functions for the project.
"""
import os
import time
import json
from typing import Any, Dict, List, Optional, Union
from datetime import datetime

from .logger import get_logger

logger = get_logger(__name__)

def get_timestamp() -> str:
    """
    Get the current timestamp in ISO format.
    
    Returns:
        Current timestamp string in ISO format
    """
    return datetime.now().isoformat()

def read_json_file(file_path: str) -> Dict[str, Any]:
    """
    Read and parse a JSON file.
    
    Args:
        file_path: Path to the JSON file
        
    Returns:
        Parsed JSON content as a dictionary
        
    Raises:
        FileNotFoundError: If the file doesn't exist
        json.JSONDecodeError: If the file is not valid JSON
    """
    logger.debug(f"Reading JSON file: {file_path}")
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
        raise
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in {file_path}: {e}")
        raise

def write_json_file(file_path: str, data: Dict[str, Any], indent: int = 4) -> None:
    """
    Write data to a JSON file.
    
    Args:
        file_path: Path to the JSON file
        data: Data to write
        indent: Indentation level for the JSON file (default: 4)
    """
    logger.debug(f"Writing JSON file: {file_path}")
    # Ensure directory exists
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=indent)

def timed_execution(func):
    """
    Decorator to measure the execution time of a function.
    
    Args:
        func: The function to measure
        
    Returns:
        Wrapper function that measures and logs execution time
    """
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        logger.debug(f"Function {func.__name__} executed in {end_time - start_time:.4f} seconds")
        return result
    return wrapper