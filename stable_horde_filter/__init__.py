"""
Stable Horde Filter - Image validation package.

This package provides functionality to validate images for quality, color richness,
authenticity, and absence of censorship content.
"""

__version__ = "0.1.0"

from stable_horde_filter.filter import validate_image

__all__ = ["validate_image"]