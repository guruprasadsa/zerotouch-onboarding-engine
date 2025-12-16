"""
Image Utilities for Document OCR Extractor.
Provides image loading, PDF conversion, and preprocessing for optimal OCR extraction.
"""

import fitz  # PyMuPDF
import cv2
import numpy as np
from PIL import Image
from typing import List, Union, Optional

# Image Loading
def load_image_from_path(path: str) -> Image.Image:
    """Load an image from a file path."""
    return Image.open(path).convert("RGB")

# PDF Conversion
def convert_pdf_to_images(pdf_path: str, zoom: float = 2.0) -> List[np.ndarray]:
    """
    Convert a PDF file to a list of images (numpy arrays, BGR format).
    """
    images = []
    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        raise RuntimeError(f"Failed to open PDF: {e}")

    for page_num in range(len(doc)):
        try:
            page = doc.load_page(page_num)
            pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom))
            img_data = pix.tobytes("png")
            nparr = np.frombuffer(img_data, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img is not None:
                images.append(img)
        except Exception:
            continue

    return images

# OCR Preprocessing Pipeline
def preprocess_image_for_ocr(
    image: Union[np.ndarray, Image.Image],
    apply_denoise: bool = False,
    enhance_contrast: bool = False,
    sharpen: bool = False,
    deskew: bool = False
) -> np.ndarray:
    """
    Preprocessing pipeline for OCR optimization.
    
    Args:
        image: Input image (numpy array or PIL Image)
        apply_denoise: Apply noise reduction
        enhance_contrast: Apply CLAHE contrast enhancement
        sharpen: Apply adaptive sharpening
        deskew: Attempt to correct image rotation
    
    Returns:
        Preprocessed grayscale image as numpy array
    """
    if isinstance(image, Image.Image):
        image = np.array(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    # Step 1: Convert to grayscale (PRIMARY - always do this)
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Optional steps - disabled by default for synthetic data
    if deskew:
        gray = _deskew_image(gray)
    
    if enhance_contrast:
        gray = _apply_clahe(gray)
    
    if apply_denoise:
        gray = _denoise_image(gray)
    
    if sharpen:
        gray = _sharpen_image(gray)
    
    return gray


def _apply_clahe(
    image: np.ndarray,
    clip_limit: float = 2.0,
    tile_grid_size: tuple = (8, 8)
) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    return clahe.apply(image)


def _denoise_image(
    image: np.ndarray,
    strength: int = 10
) -> np.ndarray:
    return cv2.fastNlMeansDenoising(image, None, strength, 7, 21)


def _sharpen_image(
    image: np.ndarray,
    amount: float = 1.0
) -> np.ndarray:
    """
    Apply unsharp masking for edge enhancement.
    
    Helps OCR distinguish character boundaries, especially
    for characters with similar shapes.
    """
    # Create Gaussian blurred version
    blurred = cv2.GaussianBlur(image, (0, 0), 3)
    
    # Unsharp mask formula: sharpened = original + amount * (original - blurred)
    sharpened = cv2.addWeighted(image, 1.0 + amount, blurred, -amount, 0)
    
    return sharpened


def _deskew_image(image: np.ndarray, max_angle: float = 10.0) -> np.ndarray:
    """
    Correct slight rotation in scanned documents.
    
    Uses Hough transform to detect dominant line angles
    and rotates to correct.
    """
    # Detect edges
    edges = cv2.Canny(image, 50, 150, apertureSize=3)
    
    # Detect lines using Hough transform
    lines = cv2.HoughLinesP(
        edges, 1, np.pi / 180, 100,
        minLineLength=100, maxLineGap=10
    )
    
    if lines is None or len(lines) == 0:
        return image
    
    # Calculate median angle of detected lines
    angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        if x2 - x1 != 0:
            angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
            if abs(angle) < max_angle:
                angles.append(angle)
    
    if not angles:
        return image
    
    median_angle = np.median(angles)
    
    # Only correct if angle is significant
    if abs(median_angle) < 0.5:
        return image
    
    # Rotate image
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, median_angle, 1.0)
    rotated = cv2.warpAffine(
        image, rotation_matrix, (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE
    )
    
    return rotated


def preprocess_for_structured_fields(
    image: Union[np.ndarray, Image.Image]
) -> np.ndarray:
    if isinstance(image, Image.Image):
        image = np.array(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Strong CLAHE for better character distinction
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
    enhanced = clahe.apply(gray)
    
    # Bilateral filter (preserves edges while smoothing)
    smooth = cv2.bilateralFilter(enhanced, 9, 75, 75)
    
    # Strong sharpening
    kernel = np.array([[-1, -1, -1],
                       [-1,  9, -1],
                       [-1, -1, -1]])
    sharpened = cv2.filter2D(smooth, -1, kernel)
    
    return sharpened


def resize_for_ocr(
    image: np.ndarray,
    target_height: int = 1000,
    max_width: int = 2000
) -> np.ndarray:
    """
    Resize image to optimal dimensions for OCR.
    """
    h, w = image.shape[:2]
    
    # Calculate scale to achieve target height
    scale = target_height / h
    new_w = int(w * scale)
    
    # Cap width to prevent very wide images
    if new_w > max_width:
        scale = max_width / w
        new_w = max_width
    
    new_h = int(h * scale)
    
    # Use INTER_CUBIC for upscaling, INTER_AREA for downscaling
    interpolation = cv2.INTER_CUBIC if scale > 1 else cv2.INTER_AREA
    
    return cv2.resize(image, (new_w, new_h), interpolation=interpolation)


def remove_borders(
    image: np.ndarray,
    border_color_thresh: int = 240,
    min_content_ratio: float = 0.5
) -> np.ndarray:
    """
    Remove white/light borders from scanned documents.
    
    Helps focus OCR on actual content area.
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Create mask of content (non-white pixels)
    _, thresh = cv2.threshold(gray, border_color_thresh, 255, cv2.THRESH_BINARY_INV)
    
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return image
    
    # Find bounding rectangle of all content
    all_points = np.vstack(contours)
    x, y, w, h = cv2.boundingRect(all_points)
    
    # Validate crop area
    img_h, img_w = image.shape[:2]
    content_ratio = (w * h) / (img_w * img_h)
    
    if content_ratio < min_content_ratio:
        return image
    
    # Add small padding
    padding = 10
    x = max(0, x - padding)
    y = max(0, y - padding)
    w = min(img_w - x, w + 2 * padding)
    h = min(img_h - y, h + 2 * padding)
    
    return image[y:y+h, x:x+w]

