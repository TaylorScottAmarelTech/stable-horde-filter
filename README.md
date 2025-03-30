# Stable Horde Filter

A robust image validation package that checks for quality, color richness, authenticity, and absence of censorship in images. This package is designed to filter out problematic or low-quality images from image generation pipelines.

## Installation

### Basic Installation

```bash
pip install stable-horde-filter
```

### Full Installation (with OCR and advanced features)

```bash
pip install stable-horde-filter[full]
```

## Usage

The package provides a single, simple function that takes an image path and returns a boolean result:

```python
from stable_horde_filter import validate_image

# Basic usage
is_valid = validate_image('/path/to/image.jpg')

# With verbose output
is_valid = validate_image('/path/to/image.jpg', verbose=True)

if is_valid:
    print("Image passed all validation checks!")
else:
    print("Image failed validation")
```

### Command Line Usage

You can also run the validation from the command line:

```bash
python -m stable_horde_filter.filter /path/to/image.jpg
```

## Validation Checks

The package performs comprehensive image validation in four stages:

1. **Strict Quality Validation**
   - Image dimensions and format
   - Brightness levels
   - Color balance
   - Corner pixel analysis
   - Local texture variation

2. **Color Richness Validation**
   - Colorfulness metrics
   - Color pixel percentage
   - Color variation analysis

3. **Authenticity Validation**
   - Solid color detection
   - Edge feature analysis
   - Pattern detection
   - Texture complexity assessment

4. **Censorship Detection**
   - Text recognition (with OCR if available)
   - Censorship pattern detection
   - Dark background with text analysis

## Dependencies

- Required:
  - numpy
  - Pillow (PIL)
  - scipy

- Optional:
  - opencv-python (for advanced pattern detection)
  - pytesseract (for OCR-based censorship detection)

## License

This project is licensed under the MIT License - see the LICENSE file for details.