import fitz  # PyMuPDF
import cv2
import numpy as np
from PIL import Image
from typing import List, Union


# --------------------------------------------------
# Image Loading
# --------------------------------------------------
def load_image_from_path(path: str) -> Image.Image:
    """Load an image from a file path."""
    return Image.open(path).convert("RGB")


# --------------------------------------------------
# PDF Conversion
# --------------------------------------------------
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
            # Skip problematic pages instead of failing whole document
            continue

    return images


# --------------------------------------------------
# OCR Preprocessing (Optional)
# --------------------------------------------------
def preprocess_image_for_ocr(
    image: Union[np.ndarray, Image.Image],
    apply_denoise: bool = True
) -> np.ndarray:
    """
    Light preprocessing suitable for EasyOCR.
    Avoids aggressive thresholding.
    """
    if isinstance(image, Image.Image):
        image = np.array(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Mild denoising (safe for CNN-based OCR)
    if apply_denoise:
        gray = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)

    return gray
