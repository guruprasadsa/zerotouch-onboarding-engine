"""
Mock ERP Client Implementation

This module provides a simulated ERP client for:
- Local development and testing
- Integration testing without real ERP access
- Demonstrating expected behavior patterns

Design Intent:
- Simulate realistic ERP behavior including validation and duplicate detection
- Support configurable failure scenarios for testing
- Maintain in-memory state for test verification
- No external dependencies
"""
import uuid
from typing import Optional
from datetime import datetime

from src.core.vendor_model import Vendor
from src.integration.erp.base import ERPClient, ERPResponse, ERPErrorCode


class MockERPClient(ERPClient):
    """
    In-memory mock ERP client for testing and development.
    
    Features:
    - Validates vendor data before "creation"
    - Detects duplicates by PAN/GSTIN/Bank Account
    - Generates realistic vendor IDs
    - Supports failure simulation for testing edge cases
    
    Usage:
        client = MockERPClient()
        response = client.create_vendor(vendor)
        if response.success:
            print(f"Created vendor: {response.vendor_id}")
    """
    
    def __init__(self, 
                 simulate_connection_error: bool = False,
                 simulate_timeout: bool = False,
                 require_gstin: bool = False):
        """
        Initialize mock ERP client.
        
        Args:
            simulate_connection_error: If True, health_check returns False
            simulate_timeout: If True, create_vendor returns TIMEOUT error
            require_gstin: If True, GSTIN is required for vendor creation
        """
        self._vendors: dict[str, dict] = {}  # vendor_id -> vendor data
        self._pan_index: dict[str, str] = {}  # PAN -> vendor_id
        self._gstin_index: dict[str, str] = {}  # GSTIN -> vendor_id
        self._account_index: dict[str, str] = {}  # account_number -> vendor_id
        
        # Simulation flags
        self._simulate_connection_error = simulate_connection_error
        self._simulate_timeout = simulate_timeout
        self._require_gstin = require_gstin
    
    def _generate_vendor_id(self) -> str:
        """Generate realistic vendor ID (format: VND-XXXXXXXX)."""
        return f"VND-{uuid.uuid4().hex[:8].upper()}"
    
    def _generate_erp_reference(self) -> str:
        """Generate ERP transaction reference number."""
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        return f"TXN-{timestamp}-{uuid.uuid4().hex[:4].upper()}"
    
    def health_check(self) -> bool:
        """Check if mock ERP is 'connected'."""
        return not self._simulate_connection_error
    
    def check_duplicate(self, vendor: Vendor) -> bool:
        """
        Check for existing vendor by PAN, GSTIN, or bank account.
        
        Returns True if any matching vendor found.
        """
        if vendor.pan and vendor.pan.upper() in self._pan_index:
            return True
        if vendor.gstin and vendor.gstin.upper() in self._gstin_index:
            return True
        if vendor.bank_details and vendor.bank_details.account_number:
            if vendor.bank_details.account_number in self._account_index:
                return True
        return False
    
    def create_vendor(self, vendor: Vendor) -> ERPResponse:
        """
        Create vendor in mock ERP.
        
        Performs:
        1. Connection check
        2. Data validation
        3. Duplicate detection
        4. Vendor creation with ID generation
        """
        # Simulate connection issues
        if self._simulate_connection_error:
            return ERPResponse.error(
                ERPErrorCode.CONNECTION_ERROR,
                "Unable to connect to ERP system"
            )
        
        if self._simulate_timeout:
            return ERPResponse.error(
                ERPErrorCode.TIMEOUT,
                "ERP request timed out after 30 seconds"
            )
        
        # Validate vendor data
        is_valid, errors = vendor.validate()
        if not is_valid:
            return ERPResponse.error(
                ERPErrorCode.VALIDATION_ERROR,
                f"Validation failed: {'; '.join(errors)}"
            )
        
        # Additional business rules
        if self._require_gstin and not vendor.gstin:
            return ERPResponse.error(
                ERPErrorCode.VALIDATION_ERROR,
                "GSTIN is required for vendor creation"
            )
        
        # Check for duplicates
        if self.check_duplicate(vendor):
            return ERPResponse.error(
                ERPErrorCode.DUPLICATE_VENDOR,
                "Vendor with same PAN/GSTIN/Bank Account already exists"
            )
        
        # Create vendor
        vendor_id = self._generate_vendor_id()
        erp_reference = self._generate_erp_reference()
        
        # Store vendor data
        vendor_dict = vendor.to_dict()
        vendor_dict["vendor_id"] = vendor_id
        vendor_dict["erp_reference"] = erp_reference
        vendor_dict["created_at"] = datetime.now().isoformat()
        
        self._vendors[vendor_id] = vendor_dict
        
        # Update indices
        if vendor.pan:
            self._pan_index[vendor.pan.upper()] = vendor_id
        if vendor.gstin:
            self._gstin_index[vendor.gstin.upper()] = vendor_id
        if vendor.bank_details and vendor.bank_details.account_number:
            self._account_index[vendor.bank_details.account_number] = vendor_id
        
        return ERPResponse.ok(
            vendor_id=vendor_id,
            message=f"Vendor '{vendor.vendor_name}' created successfully",
            erp_reference=erp_reference
        )
    
    # -------------------------
    # Test helper methods
    # -------------------------
    
    def get_vendor(self, vendor_id: str) -> Optional[dict]:
        """Retrieve vendor by ID (for test verification)."""
        return self._vendors.get(vendor_id)
    
    def get_all_vendors(self) -> list[dict]:
        """Get all created vendors (for test verification)."""
        return list(self._vendors.values())
    
    def clear(self) -> None:
        """Reset all state (for test isolation)."""
        self._vendors.clear()
        self._pan_index.clear()
        self._gstin_index.clear()
        self._account_index.clear()
    
    def vendor_count(self) -> int:
        """Get count of created vendors."""
        return len(self._vendors)
