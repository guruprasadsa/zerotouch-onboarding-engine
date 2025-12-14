from fastapi import FastAPI, UploadFile, File, HTTPException
import shutil
import os
import tempfile
import logging
import torch

from src.core.extractor import DocumentExtractor
from src.core.schema import ExtractionResult

# --------------------------------------------------
# Logging (API-level)
# --------------------------------------------------
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

# --------------------------------------------------
# App Init
# --------------------------------------------------
app = FastAPI(
    title="ZeroTouch KYC OCR Extractor API",
    version="1.0.0"
)

# --------------------------------------------------
# Initialize Extractor
# --------------------------------------------------
has_gpu = torch.cuda.is_available()
logger.info(f"GPU Detected: {has_gpu}")
extractor = DocumentExtractor(use_gpu=has_gpu)

# --------------------------------------------------
# Allowed File Types
# --------------------------------------------------
ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".pdf"}

# --------------------------------------------------
# Routes
# --------------------------------------------------
@app.post("/extract", response_model=ExtractionResult, response_model_exclude_none=True)
async def extract_document(file: UploadFile = File(...)):
    tmp_path = None

    try:
        # Validate file extension
        ext = os.path.splitext(file.filename)[1].lower()
        if ext not in ALLOWED_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type: {ext}"
            )

        # Save to temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
            shutil.copyfileobj(file.file, tmp)
            tmp_path = tmp.name

        logger.info(f"Processing file: {file.filename}")

        # Run extraction
        result_dict = extractor.extract_from_file(tmp_path)

        # Enforce schema explicitly
        return ExtractionResult(**result_dict)

    except HTTPException:
        # Let FastAPI handle HTTP errors
        raise

    except Exception as e:
        logger.error("Unhandled extraction error", exc_info=True)
        return ExtractionResult(
            document_type="UNKNOWN",
            status="failure",
            message=str(e),
            fields={}
        )

    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass


@app.get("/")
def health_check():
    return {
        "service": "ZeroTouch KYC OCR Extractor",
        "status": "running",
        "version": "1.0.0"
    }
