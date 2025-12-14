"""
Vendor Mapper - OCR Extraction Output to Vendor Model

This module provides the transformation layer between:
- Raw OCR extraction output (document-type specific fields)
- Canonical Vendor model (ERP-ready structure)

Design Intent:
- Gracefully handle missing/partial fields
- Normalize data formats (uppercase, trimming, etc.)
- Calculate aggregate confidence scores
- Support all document types (PAN, GST, CHEQUE)
"""
from typing import Optional
from datetime import datetime

from src.core.vendor_model import Vendor, BankDetails


class VendorMapper:
    """
    Maps OCR extraction results to canonical Vendor model.
    
    Handles the complexity of:
    - Different field names per document type
    - Missing or low-confidence fields
    - Data normalization
    """
    
    @staticmethod
    def _extract_field_value(fields: dict, field_name: str) -> Optional[str]:
        """
        Safely extract field value from OCR output.
        
        OCR fields are structured as: {"field_name": {"value": "...", "confidence": 0.95}}
        """
        field = fields.get(field_name)
        if field is None:
            return None
        if isinstance(field, dict):
            return field.get("value")
        return str(field) if field else None
    
    @staticmethod
    def _extract_confidence(fields: dict, field_name: str) -> float:
        """Extract confidence score for a specific field."""
        field = fields.get(field_name)
        if field is None:
            return 0.0
        if isinstance(field, dict):
            return float(field.get("confidence", 0.0))
        return 0.0
    
    @staticmethod
    def _calculate_aggregate_confidence(fields: dict) -> float:
        """
        Calculate weighted average confidence across all extracted fields.
        
        Fields with higher business importance get higher weights.
        """
        weights = {
            "name": 1.5,
            "pan_number": 2.0,
            "gstin": 2.0,
            "legal_name": 1.5,
            "account_number": 1.0,
            "ifsc_code": 1.0,
            "payee_name": 1.5,
            "address": 0.5,
        }
        
        total_weight = 0.0
        weighted_sum = 0.0
        
        for field_name, weight in weights.items():
            if field_name in fields:
                confidence = VendorMapper._extract_confidence(fields, field_name)
                weighted_sum += confidence * weight
                total_weight += weight
        
        return weighted_sum / total_weight if total_weight > 0 else 0.0
    
    @staticmethod
    def _normalize_text(value: Optional[str]) -> Optional[str]:
        """Normalize text: uppercase, strip whitespace."""
        if value is None:
            return None
        return value.strip().upper() if value.strip() else None

    def map_from_extraction(self, extraction_result: dict) -> Optional[Vendor]:
        """
        Convert OCR extraction result to Vendor model.
        
        Args:
            extraction_result: Output from DocumentExtractor.extract_from_file()
                Expected structure:
                {
                    "document_type": "PAN" | "GST" | "CHEQUE",
                    "status": "success" | "failure",
                    "fields": {...}
                }
        
        Returns:
            Vendor instance or None if extraction failed or no usable data
        """
        # Check extraction status
        if extraction_result.get("status") != "success":
            return None
        
        fields = extraction_result.get("fields", {})
        doc_type = extraction_result.get("document_type", "UNKNOWN")
        
        # Extract vendor name (varies by document type)
        vendor_name = (
            self._extract_field_value(fields, "name") or  # PAN
            self._extract_field_value(fields, "legal_name") or  # GST
            self._extract_field_value(fields, "payee_name") or  # CHEQUE
            None
        )
        
        # Cannot create vendor without a name
        if not vendor_name:
            return None
        
        # Build bank details if available
        bank_details = None
        account_number = self._extract_field_value(fields, "account_number")
        ifsc_code = self._extract_field_value(fields, "ifsc_code")
        account_holder = self._extract_field_value(fields, "payee_name")
        
        if account_number or ifsc_code:
            bank_details = BankDetails(
                account_number=account_number,
                ifsc_code=self._normalize_text(ifsc_code),
                account_holder=account_holder
            )
        
        # Create vendor instance
        return Vendor(
            vendor_name=vendor_name,
            pan=self._normalize_text(self._extract_field_value(fields, "pan_number")),
            gstin=self._normalize_text(self._extract_field_value(fields, "gstin")),
            bank_details=bank_details,
            address=self._extract_field_value(fields, "address"),
            source_document_type=doc_type,
            extraction_timestamp=datetime.now(),
            confidence_score=self._calculate_aggregate_confidence(fields)
        )
    
    def map_multiple(self, extraction_results: list[dict]) -> list[Vendor]:
        """
        Map multiple extraction results to vendor models.
        
        Useful for batch processing scenarios.
        Filters out None results from failed extractions.
        """
        vendors = []
        for result in extraction_results:
            vendor = self.map_from_extraction(result)
            if vendor:
                vendors.append(vendor)
        return vendors
