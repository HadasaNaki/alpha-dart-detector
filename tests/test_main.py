"""
Tests for the main module.
"""
import unittest
from unittest import mock

from src.main import run_application, main


class TestMain(unittest.TestCase):
    """Test cases for the main module."""
    
    @mock.patch('src.main.config')
    @mock.patch('src.main.logger')
    def test_run_application_success(self, mock_logger, mock_config):
        """Test that run_application executes successfully."""
        # Configure mocks
        mock_config.get.side_effect = lambda key, default=None: {
            'app.name': 'test_app',
            'app.version': '1.0.0',
            'features.enable_feature_a': True,
            'features.enable_feature_b': False
        }.get(key, default)
        
        # Run the function
        result = run_application()
        
        # Verify function returned success code
        self.assertEqual(result, 0)
        
        # Verify config values were accessed
        mock_config.get.assert_any_call('app.name', 'python_project')
        mock_config.get.assert_any_call('app.version', '0.1.0')
        
        # Verify logs were written
        mock_logger.info.assert_any_call('Starting test_app v1.0.0')
        mock_logger.info.assert_any_call('Feature A is enabled')
        mock_logger.info.assert_any_call('Application completed successfully')
    
    @mock.patch('src.main.run_application')
    def test_main_success(self, mock_run_app):
        """Test that main function handles success correctly."""
        # Configure mock to return success
        mock_run_app.return_value = 0
        
        # Call the main function
        result = main()
        
        # Verify the result
        self.assertEqual(result, 0)
        # Verify run_application was called
        mock_run_app.assert_called_once()
    
    @mock.patch('src.main.run_application')
    @mock.patch('src.main.logger')
    def test_main_exception(self, mock_logger, mock_run_app):
        """Test that main handles exceptions correctly."""
        # Configure mock to raise an exception
        mock_run_app.side_effect = Exception("Test error")
        
        # Call the main function
        result = main()
        
        # Verify the result is an error code
        self.assertEqual(result, 1)
        
        # Verify the exception was logged
        mock_logger.exception.assert_called_once()
        self.assertIn("Test error", mock_logger.exception.call_args[0][0])


if __name__ == "__main__":
    unittest.main()
