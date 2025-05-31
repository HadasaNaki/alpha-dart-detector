"""
Command Line Interface for the project.
"""
import argparse
import sys
from typing import List, Optional

from .config import config
from .logger import get_logger

logger = get_logger(__name__)

def parse_args(args: Optional[List[str]] = None) -> argparse.Namespace:
    """
    Parse command line arguments.
    
    Args:
        args: Command line arguments (defaults to sys.argv[1:])
        
    Returns:
        Parsed arguments as an argparse.Namespace object
    """
    parser = argparse.ArgumentParser(
        description="Python Project CLI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # General arguments
    parser.add_argument("-v", "--verbose", action="store_true", 
                      help="Enable verbose output")
    parser.add_argument("--config", type=str, default=None,
                      help="Path to a custom configuration file")
    
    # Create subparsers for different commands
    subparsers = parser.add_subparsers(dest="command", help="Commands")
    
    # Define the 'run' command
    run_parser = subparsers.add_parser("run", help="Run the application")
    run_parser.add_argument("--feature", choices=["a", "b"], default=None,
                          help="Enable a specific feature")
    
    # Define the 'config' command
    config_parser = subparsers.add_parser("config", help="Manage configuration")
    config_parser.add_argument("action", choices=["get", "set", "list"],
                             help="Configuration action to perform")
    config_parser.add_argument("key", nargs="?", help="Configuration key")
    config_parser.add_argument("value", nargs="?", help="Configuration value (for 'set' action)")
    
    return parser.parse_args(args)

def handle_run_command(args: argparse.Namespace) -> int:
    """
    Handle the 'run' command.
    
    Args:
        args: Parsed command line arguments
        
    Returns:
        Exit code (0 for success, non-zero for failure)
    """
    logger.info("Running the application")
    
    if args.feature == "a":
        config.set("features.enable_feature_a", True)
        logger.info("Feature A enabled")
    elif args.feature == "b":
        config.set("features.enable_feature_b", True)
        logger.info("Feature B enabled")
        
    from .main import main
    return main()

def handle_config_command(args: argparse.Namespace) -> int:
    """
    Handle the 'config' command.
    
    Args:
        args: Parsed command line arguments
        
    Returns:
        Exit code (0 for success, non-zero for failure)
    """
    if args.action == "list":
        print(json.dumps(config.all, indent=2))
        return 0
    
    if args.action == "get":
        if not args.key:
            logger.error("Key is required for 'get' action")
            return 1
            
        value = config.get(args.key)
        if value is None:
            logger.error(f"Key '{args.key}' not found in configuration")
            return 1
            
        print(value)
        return 0
        
    if args.action == "set":
        if not args.key:
            logger.error("Key is required for 'set' action")
            return 1
        if args.value is None:
            logger.error("Value is required for 'set' action")
            return 1
            
        # Try to parse the value as JSON, fall back to string
        try:
            import json
            value = json.loads(args.value)
        except json.JSONDecodeError:
            value = args.value
            
        config.set(args.key, value)
        config.save()
        logger.info(f"Set {args.key} = {value}")
        return 0
        
    logger.error(f"Unknown config action: {args.action}")
    return 1

def main(argv: Optional[List[str]] = None) -> int:
    """
    Main entry point for the CLI.
    
    Args:
        argv: Command line arguments (defaults to sys.argv[1:])
        
    Returns:
        Exit code (0 for success, non-zero for failure)
    """
    args = parse_args(argv)
    
    # Configure logging level based on verbosity
    if args.verbose:
        import logging
        logger.setLevel(logging.DEBUG)
        logger.debug("Verbose logging enabled")
    
    # Load custom configuration if specified
    if args.config:
        config._load_config_from_file(args.config)
        logger.debug(f"Loaded configuration from {args.config}")
    
    # Handle commands
    if args.command == "run":
        return handle_run_command(args)
    elif args.command == "config":
        return handle_config_command(args)
    else:
        logger.error(f"Unknown command: {args.command}")
        parse_args(["--help"])
        return 1

if __name__ == "__main__":
    sys.exit(main())