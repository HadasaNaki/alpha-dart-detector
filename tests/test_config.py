"""
Tests for the config module.
"""
import json
import os
import tempfile
import unittest
from unittest import mock

from src.config import Config


class TestConfig(unittest.TestCase):
    """Test cases for the Config class."""
    
    def setUp(self):
        """Set up test environment."""
        # Clear the singleton instance for each test
        Config._instance = None
        self.original_config_path = Config.CONFIG_FILE_PATH
    
    def tearDown(self):
        """Clean up after tests."""
        # Restore the original config path
        Config.CONFIG_FILE_PATH = self.original_config_path
        Config._instance = None
    
    def test_singleton_pattern(self):
        """Test that Config implements the singleton pattern correctly."""
        config1 = Config()
        config2 = Config()
        
        # Both instances should be the same object
        self.assertIs(config1, config2)
    
    def test_load_config_file_not_found(self):
        """Test behavior when config file is not found."""
        # Set up a non-existent path
        Config.CONFIG_FILE_PATH = "/path/to/nonexistent/config.json"
        
        # Create a config instance with mock for print function
        with mock.patch('builtins.print') as mock_print:
            config = Config()
            
            # The _config should be an empty dict
            self.assertEqual(config._config, {})
            
            # Check that a warning was printed
            mock_print.assert_called_once()
            self.assertIn("not found", mock_print.call_args[0][0])
    
    def test_get_config_values(self):
        """Test getting configuration values."""
        # Create a temporary config file
        test_config = {
            "app": {
                "name": "test_app",
                "version": "1.2.3"
            },
            "features": {
                "enable_feature_a": True
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp:
            json.dump(test_config, tmp)
            tmp_path = tmp.name
        
        try:
            # Set the config path to our temp file
            Config.CONFIG_FILE_PATH = tmp_path
            
            # Create a new config instance
            config = Config()
            
            # Test getting values
            self.assertEqual(config.get("app.name"), "test_app")
            self.assertEqual(config.get("app.version"), "1.2.3")
            self.assertTrue(config.get("features.enable_feature_a"))
            
            # Test getting non-existent values with default
            self.assertIsNone(config.get("nonexistent.key"))
            self.assertEqual(config.get("nonexistent.key", "default"), "default")
            
            # Test getting the entire config
            self.assertEqual(config.all, test_config)
        
        finally:
            os.remove(tmp_path)
    
    def test_set_config_values(self):
        """Test setting configuration values."""
        config = Config()
        
        # Set a simple value
        config.set("app.name", "new_name")
        self.assertEqual(config.get("app.name"), "new_name")
        
        # Set a nested value that doesn't exist
        config.set("new.nested.value", 42)
        self.assertEqual(config.get("new.nested.value"), 42)
        
        # Set a value in an existing path
        config.set("app.version", "2.0.0")
        self.assertEqual(config.get("app.version"), "2.0.0")

    @mock.patch("src.config.open", new_callable=mock.mock_open)
    def test_save_config(self, mock_open):
        """Test saving configuration."""
        config = Config()
        
        # Set some values
        config.set("app.name", "test_app")
        config.set("app.version", "1.0.0")
        
        # Save the config
        config.save()
        
        # Check that the file was opened correctly
        mock_open.assert_called_once_with(Config.CONFIG_FILE_PATH, "w", encoding="utf-8")
        
        # Check that json.dump was called with the right data
        handle = mock_open()
        self.assertEqual(handle.write.call_count, 1)


if __name__ == "__main__":
    unittest.main()
