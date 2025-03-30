"""Tests for core module."""

import unittest
from stable_horde_filter.core import HordeFilter

class TestHordeFilter(unittest.TestCase):
    """Test cases for the HordeFilter base class."""
    
    def test_initialization(self):
        """Test basic initialization of the filter."""
        filter_instance = HordeFilter()
        self.assertIsInstance(filter_instance, HordeFilter)
    
    def test_apply_not_implemented(self):
        """Test that apply method raises NotImplementedError."""
        filter_instance = HordeFilter()
        with self.assertRaises(NotImplementedError):
            filter_instance.apply({})


if __name__ == "__main__":
    unittest.main()