"""
Main module for the project.
"""
import sys
from typing import Dict, Any, Optional

from .config import config
from .logger import get_logger
from .utils import timed_execution

logger = get_logger(__name__)

@timed_execution
def run_application(params: Optional[Dict[str, Any]] = None) -> int:
    """
    Run the main application logic.
    
    Args:
        params: Optional parameters to customize application behavior
        
    Returns:
        Exit code (0 for success, non-zero for failure)
    """
    params = params or {}
    
    # Get application configuration
    app_name = config.get('app.name', 'python_project')
    app_version = config.get('app.version', '0.1.0')
    
    logger.info(f"Starting {app_name} v{app_version}")
    logger.debug(f"Running with parameters: {params}")
    
    # Check enabled features
    feature_a = config.get('features.enable_feature_a', False)
    feature_b = config.get('features.enable_feature_b', False)
    
    if feature_a:
        logger.info("Feature A is enabled")
    
    if feature_b:
        logger.info("Feature B is enabled")
    
    # Main application logic would go here
    print(f"Welcome to {app_name} v{app_version}!")
    print("This is a Python project template with CI/CD integration.")
    
    logger.info("Application completed successfully")
    return 0

def main() -> int:
    """
    Main entry point for the application.
    
    Returns:
        Exit code (0 for success, non-zero for failure)
    """
    try:
        return run_application()
    except Exception as e:
        logger.exception(f"An error occurred: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
