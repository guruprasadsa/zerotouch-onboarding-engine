from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from typing import Optional
import shutil
import os
import tempfile
import logging
import torch

from src.core.extractor import DocumentExtractor
from src.core.schema import ExtractionResult
from src.integration.vendor_mapper import VendorMapper
from src.integration.erp import MockERPClient, ERPErrorCode


# Logging (API-level)
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

# App Init
app = FastAPI(
    title="ZeroTouch KYC OCR Extractor API",
    version="1.0.0",
    description="Extract KYC data from documents and push to ERP systems"
)

# Initialize Services
has_gpu = torch.cuda.is_available()
logger.info(f"GPU Detected: {has_gpu}")
extractor = DocumentExtractor(use_gpu=has_gpu)
vendor_mapper = VendorMapper()
erp_client = MockERPClient()

# Allowed File Types
ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".pdf"}

# Response Models
class ERPPushResponse(BaseModel):
    """Response model for ERP push endpoint."""
    extraction_status: str
    document_type: str
    erp_status: str
    vendor_id: Optional[str] = None
    erp_reference: Optional[str] = None
    message: Optional[str] = None
    
    model_config = {"exclude_none": True}


# Routes
@app.post("/extract", response_model=ExtractionResult, response_model_exclude_none=True)
async def extract_document(file: UploadFile = File(...)):
    """
    Extract structured data from a KYC document.
    
    Supports: PAN Card, GST Certificate, Bank Documents (Cheque/Certificate)
    """
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


@app.post("/extract-and-push", response_model=ERPPushResponse, response_model_exclude_none=True)
async def extract_and_push_to_erp(file: UploadFile = File(...)):
    """
    Extract data from a KYC document and push to ERP as a new vendor.
    
    Flow:
    1. Extract fields from document using OCR
    2. Map extracted fields to Vendor model
    3. Validate vendor data
    4. Create vendor in ERP system
    
    Returns vendor ID and ERP reference on success.
    """
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

        logger.info(f"Processing file for ERP push: {file.filename}")

        # Step 1: Extract from document
        extraction_result = extractor.extract_from_file(tmp_path)
        doc_type = extraction_result.get("document_type", "UNKNOWN")
        
        if extraction_result.get("status") != "success":
            return ERPPushResponse(
                extraction_status="failure",
                document_type=doc_type,
                erp_status="skipped",
                message=extraction_result.get("message", "Extraction failed")
            )

        # Step 2: Map to Vendor model
        vendor = vendor_mapper.map_from_extraction(extraction_result)
        
        if vendor is None:
            return ERPPushResponse(
                extraction_status="success",
                document_type=doc_type,
                erp_status="failure",
                message="Could not map extraction to vendor (missing required fields like name)"
            )

        # Step 3: Validate vendor
        is_valid, errors = vendor.validate()
        if not is_valid:
            return ERPPushResponse(
                extraction_status="success",
                document_type=doc_type,
                erp_status="failure",
                message=f"Vendor validation failed: {'; '.join(errors)}"
            )

        # Step 4: Push to ERP
        erp_response = erp_client.create_vendor(vendor)
        
        if erp_response.success:
            return ERPPushResponse(
                extraction_status="success",
                document_type=doc_type,
                erp_status="success",
                vendor_id=erp_response.vendor_id,
                erp_reference=erp_response.erp_reference,
                message=erp_response.message
            )
        else:
            return ERPPushResponse(
                extraction_status="success",
                document_type=doc_type,
                erp_status="failure",
                message=f"ERP Error ({erp_response.error_code.value}): {erp_response.message}"
            )

    except HTTPException:
        raise

    except Exception as e:
        logger.error("Unhandled error in extract-and-push", exc_info=True)
        return ERPPushResponse(
            extraction_status="failure",
            document_type="UNKNOWN",
            erp_status="error",
            message=str(e)
        )

    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass


@app.get("/erp/vendors")
async def list_vendors():
    """
    List all vendors created in the ERP system (mock).
    
    Useful for testing and verification.
    """
    return {
        "count": erp_client.vendor_count(),
        "vendors": erp_client.get_all_vendors()
    }


@app.delete("/erp/vendors")
async def clear_vendors():
    """
    Clear all vendors from the ERP system (mock).
    
    Useful for testing.
    """
    erp_client.clear()
    return {"message": "All vendors cleared", "count": 0}


@app.get("/")
def health_check():
    return {
        "service": "ZeroTouch KYC OCR Extractor",
        "status": "running",
        "version": "1.0.0",
        "erp_connected": erp_client.health_check(),
        "gpu_enabled": has_gpu
    }
