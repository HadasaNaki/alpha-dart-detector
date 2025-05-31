"""
Tests for the utils module.
"""
import json
import os
import tempfile
import unittest
from datetime import datetime
from unittest import mock

from src.utils import get_timestamp, read_json_file, write_json_file, timed_execution


class TestUtils(unittest.TestCase):
    """Test cases for utility functions."""

    def test_get_timestamp(self):
        """Test the get_timestamp function returns a valid ISO format date."""
        timestamp = get_timestamp()
        
        # Verify the timestamp can be parsed as a datetime object
        try:
            dt = datetime.fromisoformat(timestamp)
            self.assertIsInstance(dt, datetime)
        except ValueError:
            self.fail("get_timestamp did not return a valid ISO format datetime string")

    def test_read_json_file(self):
        """Test reading JSON files."""
        # Create a temporary JSON file
        test_data = {"key": "value", "number": 42}
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp:
            json.dump(test_data, tmp)
            tmp_path = tmp.name
        
        try:
            # Test reading the file
            result = read_json_file(tmp_path)
            self.assertEqual(result, test_data)
            
            # Test reading non-existent file
            with self.assertRaises(FileNotFoundError):
                read_json_file("non_existent_file.json")
                
            # Test reading invalid JSON
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as inv:
                inv.write("invalid json{")
                inv_path = inv.name
                
            with self.assertRaises(json.JSONDecodeError):
                read_json_file(inv_path)
                
            os.remove(inv_path)
                
        finally:
            os.remove(tmp_path)

    def test_write_json_file(self):
        """Test writing JSON files."""
        test_data = {"key": "value", "nested": {"a": 1, "b": 2}}
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Test writing to a new file
            file_path = os.path.join(tmp_dir, "test_file.json")
            write_json_file(file_path, test_data)
            
            # Verify the file was written correctly
            with open(file_path, 'r') as f:
                content = json.load(f)
                self.assertEqual(content, test_data)
                
            # Test writing to a nested directory that doesn't exist
            nested_path = os.path.join(tmp_dir, "subdir", "nested", "file.json")
            write_json_file(nested_path, test_data)
            
            # Verify the file was created with directories
            self.assertTrue(os.path.exists(nested_path))
            with open(nested_path, 'r') as f:
                content = json.load(f)
                self.assertEqual(content, test_data)

    def test_timed_execution_decorator(self):
        """Test the timed_execution decorator."""
        
        # Mock function to decorate
        @timed_execution
        def test_func(x, y):
            return x + y
        
        # Mock the logging
        with mock.patch('src.utils.logger') as mock_logger:
            # Call the function
            result = test_func(3, 5)
            
            # Verify the function returns the correct result
            self.assertEqual(result, 8)
            
            # Verify that logger.debug was called
            mock_logger.debug.assert_called_once()
            # The call should contain the function name and execution time
            call_args = mock_logger.debug.call_args[0][0]
            self.assertIn('test_func', call_args)
            self.assertIn('executed in', call_args)


if __name__ == '__main__':
    unittest.main()