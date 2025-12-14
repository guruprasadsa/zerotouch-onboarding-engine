"""
Canonical Vendor Data Model

This module defines the vendor data structure that is independent of:
- OCR extraction output format
- ERP system-specific schemas

Design Intent:
- Single source of truth for vendor data within the application
- Validated at construction time using Pydantic
- Immutable after creation for data integrity
"""
from dataclasses import dataclass, field
from typing import Optional
from datetime import datetime
import re


@dataclass(frozen=True)
class BankDetails:
    """Bank account information for vendor payments."""
    account_number: Optional[str] = None
    ifsc_code: Optional[str] = None
    account_holder: Optional[str] = None
    
    def is_complete(self) -> bool:
        """Check if minimum required bank details are present."""
        return bool(self.account_number and self.ifsc_code)


@dataclass(frozen=True)
class Vendor:
    """
    Canonical vendor representation.
    
    This model captures all fields needed for vendor master creation
    across different ERP systems. Fields are optional to support
    partial extraction scenarios.
    """
    # Core identification
    vendor_name: str
    
    # Tax identifiers
    pan: Optional[str] = None
    gstin: Optional[str] = None
    
    # Banking
    bank_details: Optional[BankDetails] = None
    
    # Address
    address: Optional[str] = None
    
    # Metadata
    source_document_type: Optional[str] = None
    extraction_timestamp: datetime = field(default_factory=datetime.now)
    confidence_score: float = 0.0
    
    def validate(self) -> tuple[bool, list[str]]:
        """
        Validate vendor data for ERP submission.
        
        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []
        
        # Required field
        if not self.vendor_name or len(self.vendor_name.strip()) < 2:
            errors.append("vendor_name is required and must be at least 2 characters")
        
        # PAN format validation (if provided)
        if self.pan:
            if not re.match(r'^[A-Z]{5}[0-9]{4}[A-Z]$', self.pan.upper()):
                errors.append(f"Invalid PAN format: {self.pan}")
        
        # GSTIN format validation (if provided)
        if self.gstin:
            if not re.match(r'^\d{2}[A-Z]{5}\d{4}[A-Z][A-Z0-9]Z[A-Z0-9]$', self.gstin.upper()):
                errors.append(f"Invalid GSTIN format: {self.gstin}")
        
        # IFSC format validation (if bank details provided)
        if self.bank_details and self.bank_details.ifsc_code:
            if not re.match(r'^[A-Z]{4}0[A-Z0-9]{6}$', self.bank_details.ifsc_code.upper()):
                errors.append(f"Invalid IFSC format: {self.bank_details.ifsc_code}")
        
        return (len(errors) == 0, errors)
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "vendor_name": self.vendor_name,
            "pan": self.pan,
            "gstin": self.gstin,
            "bank_account": self.bank_details.account_number if self.bank_details else None,
            "ifsc": self.bank_details.ifsc_code if self.bank_details else None,
            "account_holder": self.bank_details.account_holder if self.bank_details else None,
            "address": self.address,
            "source_document_type": self.source_document_type,
            "extraction_timestamp": self.extraction_timestamp.isoformat(),
            "confidence_score": self.confidence_score
        }
