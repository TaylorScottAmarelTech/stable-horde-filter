"""Tests for filter module."""

import unittest
from stable_horde_filter.filter import CustomFilter

class TestCustomFilter(unittest.TestCase):
    """Test cases for the CustomFilter implementation."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.filter = CustomFilter()
        self.test_data = {"sample": "data"}
    
    def test_initialization_with_options(self):
        """Test initialization with custom options."""
        options = {"option1": "value1"}
        filter_with_options = CustomFilter(options=options)
        self.assertEqual(filter_with_options.options, options)
    
    def test_apply_returns_data(self):
        """Test that apply method returns the input data."""
        result = self.filter.apply(self.test_data)
        self.assertEqual(result, self.test_data)


if __name__ == "__main__":
    unittest.main()