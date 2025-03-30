"""Core functionality for stable-horde-filter."""

# Implement core functionality here
class HordeFilter:
    """Base class for Stable Horde filters."""
    
    def __init__(self):
        """Initialize the filter."""
        pass
    
    def apply(self, data):
        """Apply the filter to the provided data."""
        raise NotImplementedError("Subclasses must implement apply method")