"""
Stable Horde Filter - Image Validation Module

This module provides functionality to validate images for quality, color richness,
authenticity, and absence of censorship content.
"""

import os
import logging
import numpy as np
import colorsys
from PIL import Image
from scipy import ndimage
from dataclasses import dataclass
from typing import Dict, Any, Tuple, Optional


@dataclass
class ValidationResult:
    """Data class for storing validation results."""
    passed: bool
    validator_name: str
    details: Dict[str, Any] = None
    error_message: Optional[str] = None


def validate_image(image_path: str, verbose: bool = False) -> bool:
    """
    Validates an image for quality, color richness, authenticity, and absence of censorship.
    
    Args:
        image_path (str): Path to the image file
        verbose (bool, optional): Whether to print detailed validation results. Defaults to False.
        
    Returns:
        bool: True if the image passes all validations, False otherwise
    """
    logger = logging.getLogger(__name__)
    
    if verbose:
        # Configure logging if verbose mode is enabled
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            force=True
        )
    
    # Check if file exists
    if not os.path.exists(image_path):
        if verbose:
            print_status(f"Image file not found: {image_path}", "error")
        return False
    
    try:
        # Common preprocessing: load image once for all validations
        img_pil = open_and_validate_image_format(image_path)
        
        if img_pil is None:
            if verbose:
                print_status("Failed to open image or invalid image format", "error")
            return False
            
        img_array = np.array(img_pil)
        
        # Stage 1: Strict image validation
        if verbose:
            print_status("Stage 1: Performing strict image validation", "step")
            
        strict_passed = validate_strict_quality(img_array, verbose=verbose)
        if not strict_passed:
            if verbose:
                print_status("Strict image validation failed", "error")
            return False
        
        # Stage 2: Color validation
        if verbose:
            print_status("Stage 2: Performing color richness validation", "step")
            
        color_passed, color_error = validate_color_richness(img_array, verbose=verbose)
        if not color_passed:
            if verbose:
                print_status(f"Color validation failed: {color_error}", "error")
            return False
        
        # Stage 3: Authenticity validation
        if verbose:
            print_status("Stage 3: Performing authenticity validation", "step")
            
        auth_passed, auth_error = validate_authenticity(img_array, verbose=verbose)
        if not auth_passed:
            if verbose:
                print_status(f"Authenticity validation failed: {auth_error}", "error")
            return False
        
        # Stage 4: Censorship detection
        if verbose:
            print_status("Stage 4: Checking for censorship content", "step")
            
        censorship_passed, censorship_reason = validate_no_censorship(
            img_array, image_path, verbose=verbose
        )
        if not censorship_passed:
            if verbose:
                print_status(f"Censorship content detected: {censorship_reason}", "error")
            return False
        
        # All validations passed
        if verbose:
            print_status("Image passed all validation checks", "success")
        return True
        
    except Exception as e:
        if verbose:
            print_status(f"Error during image validation: {str(e)}", "error")
        logger.error(f"Error validating image: {str(e)}")
        return False


def open_and_validate_image_format(image_path: str) -> Optional[Image.Image]:
    """
    Opens an image and validates its format.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        Optional[Image.Image]: PIL Image object if valid, None otherwise
    """
    try:
        with Image.open(image_path) as img:
            # Check image format
            format_type = img.format
            allowed_formats = ['PNG', 'JPEG', 'JPG', 'WEBP']
            if format_type not in allowed_formats:
                return None
                
            # Check image mode
            if img.mode != 'RGB':
                img = img.convert('RGB')
                
            # Check dimensions
            width, height = img.size
            min_size = (32, 32)
            max_size = (8192, 8192)
            
            if not (min_size[0] <= width <= max_size[0] and 
                    min_size[1] <= height <= max_size[1]):
                return None
                
            # Return a copy of the image
            return img.copy()
    except Exception:
        return None


def validate_strict_quality(img_array: np.ndarray, verbose: bool = False) -> bool:
    """
    Performs strict image quality validation.
    
    Args:
        img_array: NumPy array of the image
        verbose: Whether to print detailed validation results
        
    Returns:
        bool: True if the image passes validation, False otherwise
    """
    try:
        # Check for valid dimensions
        if img_array.ndim != 3 or img_array.shape[2] != 3:
            return False
            
        # Check for dark images
        mean_brightness = np.mean(img_array)
        if mean_brightness < 70:
            if verbose:
                print_status(f"Image is too dark (brightness: {mean_brightness:.1f})", "warning")
            return False
            
        # Check for black pixels
        black_pixels = np.all(img_array < 25, axis=2)
        black_percentage = np.mean(black_pixels)
        if black_percentage > 0.6:
            if verbose:
                print_status(f"Image contains too many black pixels ({black_percentage:.1%})", "warning")
            return False
            
        # Check for blue/purple dominance (commonly seen in AI-generated images)
        avg_r = np.mean(img_array[:, :, 0])
        avg_g = np.mean(img_array[:, :, 1])
        avg_b = np.mean(img_array[:, :, 2])
        
        total_brightness = avg_r + avg_g + avg_b
        if total_brightness > 0:
            blue_dominance = avg_b / total_brightness
            if blue_dominance > 0.5:
                if verbose:
                    print_status(f"Image has excessive blue dominance ({blue_dominance:.2f})", "warning")
                return False
                
        # Check for grayscale or low saturation
        rgb_normalized = img_array.astype(float) / 255.0
        hsv_colors = np.array([colorsys.rgb_to_hsv(r, g, b) 
                              for r, g, b in rgb_normalized.reshape(-1, 3)])
        saturations = hsv_colors[:, 1]
        
        if np.mean(saturations) < 0.15:
            if verbose:
                print_status(f"Image has very low saturation ({np.mean(saturations):.2f})", "warning")
            return False
            
        # Check corner pixels (detect vignetting or centering issues)
        height, width = img_array.shape[:2]
        corner_size = max(5, int(min(height, width) * 0.05))
        
        corners = {
            'top_left': img_array[:corner_size, :corner_size],
            'top_right': img_array[:corner_size, -corner_size:],
            'bottom_left': img_array[-corner_size:, :corner_size],
            'bottom_right': img_array[-corner_size:, -corner_size:]
        }
        
        dark_corners = 0
        for corner in corners.values():
            if np.mean(corner) < 30:
                dark_corners += 1
                
        if dark_corners > 2:
            if verbose:
                print_status(f"Image has {dark_corners} dark corners", "warning")
            return False
            
        # Check local variance (detect artificial patterns)
        gray = np.mean(img_array, axis=2)
        local_var = ndimage.generic_filter(gray, np.var, size=3)
        if np.mean(local_var) < 12.0:
            if verbose:
                print_status(f"Image lacks natural texture variation ({np.mean(local_var):.2f})", "warning")
            return False
            
        return True
        
    except Exception as e:
        logging.error(f"Error in strict quality validation: {str(e)}")
        return False


def validate_color_richness(img_array: np.ndarray, verbose: bool = False,
                           min_colorfulness: float = 14.0, 
                           min_color_percentage: float = 14.0) -> Tuple[bool, Optional[str]]:
    """
    Validates if an image has sufficient color richness.
    
    Args:
        img_array: NumPy array of the image
        verbose: Whether to print detailed validation results
        min_colorfulness: Minimum colorfulness score required
        min_color_percentage: Minimum percentage of pixels that should have significant color
        
    Returns:
        tuple: (passed, error_message)
    """
    try:
        # Convert to float for calculations
        r = img_array[:, :, 0].astype(np.float32)
        g = img_array[:, :, 1].astype(np.float32)
        b = img_array[:, :, 2].astype(np.float32)
        
        # Compute rg = R - G
        rg = np.abs(r - g)
        
        # Compute yb = 0.5 * (R + G) - B
        yb = np.abs(0.5 * (r + g) - b)
        
        # Compute the mean and standard deviation of rg and yb
        rg_mean = np.mean(rg)
        rg_std = np.std(rg)
        yb_mean = np.mean(yb)
        yb_std = np.std(yb)
        
        # Combine the mean and standard deviations
        std_root = np.sqrt((rg_std ** 2) + (yb_std ** 2))
        mean_root = np.sqrt((rg_mean ** 2) + (yb_mean ** 2))
        
        # Calculate colorfulness metric
        colorfulness = std_root + (0.3 * mean_root)
        
        # Calculate color percentage
        max_rgb = np.maximum(np.maximum(r, g), b)
        min_rgb = np.minimum(np.minimum(r, g), b)
        rgb_diff = max_rgb - min_rgb
        
        # Count pixels with significant color (threshold = 15)
        colorful_pixels = np.sum(rgb_diff > 15)
        total_pixels = img_array.shape[0] * img_array.shape[1]
        
        # Calculate percentage
        color_percentage = (colorful_pixels / total_pixels) * 100
        
        if verbose:
            print_status(f"Colorfulness score: {colorfulness:.2f} (min: {min_colorfulness})", "info")
            print_status(f"Color percentage: {color_percentage:.2f}% (min: {min_color_percentage}%)", "info")
        
        # Check if image passes both color criteria
        if colorfulness < min_colorfulness:
            return False, f"Image lacks sufficient color variation (score: {colorfulness:.2f})"
        
        if color_percentage < min_color_percentage:
            return False, f"Image has too few colored pixels ({color_percentage:.2f}%)"
        
        return True, None
        
    except Exception as e:
        return False, f"Error during color validation: {str(e)}"


def validate_authenticity(img_array: np.ndarray, verbose: bool = False,
                          max_size: int = 800) -> Tuple[bool, Optional[str]]:
    """
    Validates image authenticity by checking for artificial patterns.
    
    Args:
        img_array: NumPy array of the image
        verbose: Whether to print detailed validation results
        max_size: Maximum dimension for image processing
        
    Returns:
        tuple: (passed, error_message)
    """
    try:
        # Resize large images for faster processing
        h, w = img_array.shape[:2]
        
        # Calculate grayscale once
        gray = np.mean(img_array, axis=2)
        
        # Fast check: Solid color detection
        r_std = np.std(img_array[:,:,0])
        g_std = np.std(img_array[:,:,1])
        b_std = np.std(img_array[:,:,2])
        mean_std = (r_std + g_std + b_std) / 3.0
        
        if verbose:
            print_status(f"Color variation (std dev): {mean_std:.2f}", "info")
            
        if mean_std < 12.0:
            return False, f"Image appears to be a solid color (std dev: {mean_std:.2f})"
        
        # Fast check: Edge detection
        sobel_x = np.abs(np.gradient(gray, axis=1))
        sobel_y = np.abs(np.gradient(gray, axis=0))
        sobel_mag = np.sqrt(sobel_x**2 + sobel_y**2)
        
        # Dynamic threshold based on image content
        edge_threshold = np.mean(sobel_mag) * 2.0
        edges = sobel_mag > edge_threshold
        
        # Edge percentage check
        edge_percentage = np.mean(edges) * 100
        
        if verbose:
            print_status(f"Edge percentage: {edge_percentage:.2f}%", "info")
            
        if edge_percentage < 5.0:
            return False, f"Image has insufficient edge features ({edge_percentage:.2f}%)"
        
        # Check directional bias (common in AI generation)
        x_diff = np.mean(np.abs(sobel_x))
        y_diff = np.mean(np.abs(sobel_y))
        
        # Avoid division by zero
        if max(x_diff, y_diff) > 0:
            ratio = min(x_diff, y_diff) / max(x_diff, y_diff)
            
            if verbose:
                print_status(f"Directional gradient ratio: {ratio:.2f}", "info")
                
            if ratio < 0.3:
                direction = "vertical" if x_diff > y_diff else "horizontal"
                return False, f"Image shows strong {direction} patterns (ratio: {ratio:.2f})"
        
        # Texture analysis (simplified)
        window_size = min(32, min(h, w) // 4)
        if min(h, w) >= 64:
            step_size = window_size
            texture_samples = []
            
            for i in range(0, h - window_size, step_size):
                for j in range(0, w - window_size, step_size):
                    window = gray[i:i+window_size, j:j+window_size]
                    window_std = np.std(window)
                    texture_samples.append(window_std)
            
            if texture_samples:
                avg_texture = np.mean(texture_samples)
                
                if verbose:
                    print_status(f"Texture complexity: {avg_texture:.2f}", "info")
                    
                if avg_texture < 6.0:
                    return False, f"Image lacks texture complexity (avg: {avg_texture:.2f})"
        
        # All checks passed
        return True, None
        
    except Exception as e:
        return False, f"Error during authenticity validation: {str(e)}"


def validate_no_censorship(img_array: np.ndarray, image_path: str, 
                           verbose: bool = False) -> Tuple[bool, Optional[str]]:
    """
    Validates that the image doesn't contain censorship indicators.
    
    Args:
        img_array: NumPy array of the image
        image_path: Path to the image file (needed for optional OCR)
        verbose: Whether to print detailed validation results
        
    Returns:
        tuple: (passed, error_message)
    """
    try:
        # Define censorship keywords to detect
        CENSORSHIP_KEYWORDS = ["CENSORED", "NSFW", "CSAM", "CONTENT", "DETECTED", "BLOCKED"]
        
        # Convert to grayscale
        gray = np.mean(img_array, axis=2).astype(np.uint8)
        
        # Check for dark background (common in censorship images)
        is_dark_background = np.mean(gray) < 50
        
        # Try OCR detection if pytesseract is available
        has_censorship_text = False
        reason = None
        
        try:
            import pytesseract
            text_data = pytesseract.image_to_data(image_path, output_type=pytesseract.Output.DICT)
            
            # Extract text with reasonable confidence
            text_content = " ".join([
                text_data['text'][i].strip().upper() 
                for i in range(len(text_data['text'])) 
                if int(text_data['conf'][i]) > 60 and text_data['text'][i].strip()
            ])
            
            # Check for censorship keywords
            for keyword in CENSORSHIP_KEYWORDS:
                if keyword in text_content:
                    has_censorship_text = True
                    reason = f"Contains censorship text: '{keyword}'"
                    break
                    
                # Also check for partial matches (e.g., "CENSOR" instead of "CENSORED")
                if len(keyword) > 4 and keyword[:4] in text_content:
                    has_censorship_text = True
                    reason = f"Contains partial censorship text (likely '{keyword}')"
                    break
                    
            if verbose and text_content:
                print_status(f"OCR detected text: {text_content[:100]}", "info")
                
        except ImportError:
            # If pytesseract not available, we'll rely on pattern detection
            if verbose:
                print_status("OCR detection not available, using pattern detection", "info")
        
        # Pattern detection for censorship indicators
        if is_dark_background and not has_censorship_text:
            # Calculate percentage of the image that is very dark
            dark_ratio = np.sum(gray < 30) / gray.size
            
            # If mostly black with possible text patterns, likely censorship
            if dark_ratio > 0.7:
                # Look for bright elements on dark background
                _, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
                text_like_components = np.sum(binary > 0) / binary.size
                
                if text_like_components > 0.01 and text_like_components < 0.2:
                    has_censorship_text = True
                    reason = "Detected censorship pattern (dark background with possible text)"
        
        # Return result
        if has_censorship_text:
            return False, reason
        else:
            return True, None
            
    except Exception as e:
        # If any error occurs during detection, let it pass
        # We don't want to reject images just because the censorship detector failed
        logging.warning(f"Error in censorship detection: {str(e)}")
        return True, None


def print_status(message, status_type="info"):
    """Helper function to print colored status messages."""
    status_colors = {
        "info": "\033[94m",
        "success": "\033[92m",
        "error": "\033[91m", 
        "warning": "\033[93m",
        "step": "\033[96m",
    }
    end_color = "\033[0m"
    
    prefix = {
        "info": "ℹ️ ",
        "success": "✅ ",
        "error": "❌ ", 
        "warning": "⚠️ ",
        "step": "🔍 ",
    }
    
    print(f"{status_colors.get(status_type, '')}{prefix.get(status_type, '')}{message}{end_color}")


# Import conditional dependencies
try:
    import cv2
except ImportError:
    # Create a minimal replacement for basic functionality
    class cv2:
        @staticmethod
        def threshold(img, thresh, maxval, type):
            return None, (img > thresh) * maxval
        
        THRESH_BINARY = 0


if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) > 1:
        image_file = sys.argv[1]
        result = validate_image(image_file, verbose=True)
        print(f"\nFinal validation result: {'PASSED' if result else 'FAILED'}")
    else:
        print("Usage: python -m stable_horde_filter.filter <image_path>")