"""
ERP Client Interface

This module defines the abstract interface for ERP vendor management operations.

Design Intent:
- Decouple business logic from specific ERP implementations
- Enable testability through mock implementations
- Support multiple ERP backends (SAP, Oracle, custom REST APIs)
- Standardize error handling across implementations
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional
from enum import Enum

from src.core.vendor_model import Vendor


class ERPErrorCode(Enum):
    """Standardized error codes for ERP operations."""
    SUCCESS = "SUCCESS"
    VALIDATION_ERROR = "VALIDATION_ERROR"
    DUPLICATE_VENDOR = "DUPLICATE_VENDOR"
    CONNECTION_ERROR = "CONNECTION_ERROR"
    AUTHENTICATION_ERROR = "AUTHENTICATION_ERROR"
    PERMISSION_DENIED = "PERMISSION_DENIED"
    INTERNAL_ERROR = "INTERNAL_ERROR"
    TIMEOUT = "TIMEOUT"


@dataclass(frozen=True)
class ERPResponse:
    """
    Standardized response from ERP operations.
    
    All ERP client implementations return this structure
    regardless of underlying ERP system.
    """
    success: bool
    error_code: ERPErrorCode
    message: str
    vendor_id: Optional[str] = None
    erp_reference: Optional[str] = None  # ERP-specific reference number
    
    @classmethod
    def ok(cls, vendor_id: str, message: str = "Vendor created successfully", 
           erp_reference: Optional[str] = None) -> "ERPResponse":
        """Factory for successful responses."""
        return cls(
            success=True,
            error_code=ERPErrorCode.SUCCESS,
            message=message,
            vendor_id=vendor_id,
            erp_reference=erp_reference
        )
    
    @classmethod
    def error(cls, code: ERPErrorCode, message: str) -> "ERPResponse":
        """Factory for error responses."""
        return cls(
            success=False,
            error_code=code,
            message=message,
            vendor_id=None,
            erp_reference=None
        )


class ERPClient(ABC):
    """
    Abstract base class for ERP vendor management clients.
    
    Implementations should handle:
    - Connection management
    - Authentication
    - Data transformation to ERP-specific format
    - Error mapping to standardized codes
    """
    
    @abstractmethod
    def create_vendor(self, vendor: Vendor) -> ERPResponse:
        """
        Create a new vendor in the ERP system.
        
        Args:
            vendor: Validated Vendor model instance
            
        Returns:
            ERPResponse with vendor_id on success, error details on failure
        """
        pass
    
    @abstractmethod
    def check_duplicate(self, vendor: Vendor) -> bool:
        """
        Check if vendor already exists in ERP.
        
        Implementations should check by PAN, GSTIN, or bank account
        depending on what's available.
        
        Returns:
            True if duplicate exists, False otherwise
        """
        pass
    
    @abstractmethod
    def health_check(self) -> bool:
        """
        Verify connection to ERP system.
        
        Returns:
            True if ERP is reachable and authenticated
        """
        pass
