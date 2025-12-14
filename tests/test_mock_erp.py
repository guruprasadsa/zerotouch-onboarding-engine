"""
Unit Tests for MockERPClient

Tests the mock ERP client implementation including:
- Vendor creation
- Validation
- Duplicate detection
- Error simulation
"""
import pytest

from src.core.vendor_model import Vendor, BankDetails
from src.integration.erp.mock_erp import MockERPClient
from src.integration.erp.base import ERPErrorCode


# Fixtures  
@pytest.fixture
def erp_client():
    """Create a fresh MockERPClient instance."""
    client = MockERPClient()
    yield client
    client.clear()  # Cleanup after test


@pytest.fixture
def valid_vendor():
    """Create a valid vendor for testing."""
    return Vendor(
        vendor_name="Test Vendor Ltd",
        pan="ABCDE1234F",
        gstin="27ABCDE1234F1Z5",
        bank_details=BankDetails(
            account_number="123456789012",
            ifsc_code="HDFC0001234",
            account_holder="Test Vendor"
        ),
        address="123 Test Street, Mumbai 400001"
    )


@pytest.fixture
def vendor_without_pan():
    """Vendor without PAN number."""
    return Vendor(
        vendor_name="No PAN Vendor",
        gstin="27XYZAB5678C1Z9"
    )


@pytest.fixture
def vendor_with_invalid_pan():
    """Vendor with invalid PAN format."""
    return Vendor(
        vendor_name="Invalid PAN Vendor",
        pan="INVALID123"  # Wrong format
    )


class TestVendorCreation:
    """Tests for successful vendor creation scenarios."""
    
    def test_creates_vendor_successfully(self, erp_client, valid_vendor):
        """Should create vendor and return success."""
        response = erp_client.create_vendor(valid_vendor)
        
        assert response.success is True
        assert response.error_code == ERPErrorCode.SUCCESS
        assert response.vendor_id is not None
        assert response.vendor_id.startswith("VND-")
    
    def test_returns_erp_reference(self, erp_client, valid_vendor):
        """Should return ERP transaction reference."""
        response = erp_client.create_vendor(valid_vendor)
        
        assert response.erp_reference is not None
        assert response.erp_reference.startswith("TXN-")
    
    def test_vendor_is_stored(self, erp_client, valid_vendor):
        """Created vendor should be retrievable."""
        response = erp_client.create_vendor(valid_vendor)
        
        stored_vendor = erp_client.get_vendor(response.vendor_id)
        assert stored_vendor is not None
        assert stored_vendor["vendor_name"] == "Test Vendor Ltd"
    
    def test_vendor_count_increases(self, erp_client, valid_vendor, vendor_without_pan):
        """Vendor count should increase with each creation."""
        assert erp_client.vendor_count() == 0
        
        erp_client.create_vendor(valid_vendor)
        assert erp_client.vendor_count() == 1
        
        erp_client.create_vendor(vendor_without_pan)
        assert erp_client.vendor_count() == 2
    
    def test_creates_vendor_without_bank_details(self, erp_client):
        """Should create vendor without bank details."""
        vendor = Vendor(vendor_name="Simple Vendor", pan="ZZZZZ9999Z")
        
        response = erp_client.create_vendor(vendor)
        assert response.success is True

class TestValidationErrors:
    """Tests for validation error handling."""
    
    def test_rejects_empty_vendor_name(self, erp_client):
        """Should reject vendor with empty name."""
        vendor = Vendor(vendor_name="")
        
        response = erp_client.create_vendor(vendor)
        
        assert response.success is False
        assert response.error_code == ERPErrorCode.VALIDATION_ERROR
        assert "vendor_name" in response.message.lower()
    
    def test_rejects_invalid_pan_format(self, erp_client, vendor_with_invalid_pan):
        """Should reject vendor with invalid PAN format."""
        response = erp_client.create_vendor(vendor_with_invalid_pan)
        
        assert response.success is False
        assert response.error_code == ERPErrorCode.VALIDATION_ERROR
        assert "PAN" in response.message
    
    def test_rejects_invalid_gstin_format(self, erp_client):
        """Should reject vendor with invalid GSTIN format."""
        vendor = Vendor(
            vendor_name="Invalid GSTIN Vendor",
            gstin="INVALID_GSTIN"
        )
        
        response = erp_client.create_vendor(vendor)
        
        assert response.success is False
        assert response.error_code == ERPErrorCode.VALIDATION_ERROR
        assert "GSTIN" in response.message
    
    def test_rejects_invalid_ifsc_format(self, erp_client):
        """Should reject vendor with invalid IFSC format."""
        vendor = Vendor(
            vendor_name="Invalid IFSC Vendor",
            bank_details=BankDetails(
                account_number="123456789012",
                ifsc_code="INVALID"
            )
        )
        
        response = erp_client.create_vendor(vendor)
        
        assert response.success is False
        assert response.error_code == ERPErrorCode.VALIDATION_ERROR
        assert "IFSC" in response.message


class TestDuplicateDetection:
    """Tests for duplicate vendor detection."""
    
    def test_detects_duplicate_pan(self, erp_client, valid_vendor):
        """Should detect duplicate by PAN."""
        # Create first vendor
        erp_client.create_vendor(valid_vendor)
        
        # Try to create duplicate
        duplicate = Vendor(
            vendor_name="Different Name",
            pan=valid_vendor.pan  # Same PAN
        )
        
        response = erp_client.create_vendor(duplicate)
        
        assert response.success is False
        assert response.error_code == ERPErrorCode.DUPLICATE_VENDOR
    
    def test_detects_duplicate_gstin(self, erp_client, valid_vendor):
        """Should detect duplicate by GSTIN."""
        erp_client.create_vendor(valid_vendor)
        
        duplicate = Vendor(
            vendor_name="Different Name",
            gstin=valid_vendor.gstin  # Same GSTIN
        )
        
        response = erp_client.create_vendor(duplicate)
        
        assert response.success is False
        assert response.error_code == ERPErrorCode.DUPLICATE_VENDOR
    
    def test_detects_duplicate_account_number(self, erp_client, valid_vendor):
        """Should detect duplicate by bank account number."""
        erp_client.create_vendor(valid_vendor)
        
        duplicate = Vendor(
            vendor_name="Different Name",
            bank_details=BankDetails(
                account_number=valid_vendor.bank_details.account_number,  # Same account
                ifsc_code="ICIC0001111"
            )
        )
        
        response = erp_client.create_vendor(duplicate)
        
        assert response.success is False
        assert response.error_code == ERPErrorCode.DUPLICATE_VENDOR
    
    def test_check_duplicate_returns_true_for_existing(self, erp_client, valid_vendor):
        """check_duplicate should return True for existing vendor."""
        erp_client.create_vendor(valid_vendor)
        
        assert erp_client.check_duplicate(valid_vendor) is True
    
    def test_check_duplicate_returns_false_for_new(self, erp_client, valid_vendor):
        """check_duplicate should return False for new vendor."""
        assert erp_client.check_duplicate(valid_vendor) is False

class TestErrorSimulation:
    """Tests for simulated error scenarios."""
    
    def test_simulates_connection_error(self, valid_vendor):
        """Should simulate connection error when configured."""
        client = MockERPClient(simulate_connection_error=True)
        
        response = client.create_vendor(valid_vendor)
        
        assert response.success is False
        assert response.error_code == ERPErrorCode.CONNECTION_ERROR
    
    def test_simulates_timeout(self, valid_vendor):
        """Should simulate timeout when configured."""
        client = MockERPClient(simulate_timeout=True)
        
        response = client.create_vendor(valid_vendor)
        
        assert response.success is False
        assert response.error_code == ERPErrorCode.TIMEOUT
    
    def test_health_check_fails_on_connection_error(self):
        """Health check should fail when connection error is simulated."""
        client = MockERPClient(simulate_connection_error=True)
        
        assert client.health_check() is False
    
    def test_health_check_succeeds_normally(self, erp_client):
        """Health check should succeed under normal conditions."""
        assert erp_client.health_check() is True
    
    def test_requires_gstin_when_configured(self):
        """Should require GSTIN when configured."""
        client = MockERPClient(require_gstin=True)
        
        vendor = Vendor(
            vendor_name="No GSTIN Vendor",
            pan="ABCDE1234F"
        )
        
        response = client.create_vendor(vendor)
        
        assert response.success is False
        assert response.error_code == ERPErrorCode.VALIDATION_ERROR
        assert "GSTIN" in response.message

class TestStateManagement:
    """Tests for client state management."""
    
    def test_clear_removes_all_vendors(self, erp_client, valid_vendor):
        """clear() should remove all vendors."""
        erp_client.create_vendor(valid_vendor)
        assert erp_client.vendor_count() == 1
        
        erp_client.clear()
        assert erp_client.vendor_count() == 0
    
    def test_get_all_vendors_returns_list(self, erp_client, valid_vendor, vendor_without_pan):
        """get_all_vendors should return all created vendors."""
        erp_client.create_vendor(valid_vendor)
        erp_client.create_vendor(vendor_without_pan)
        
        vendors = erp_client.get_all_vendors()
        
        assert len(vendors) == 2
        names = [v["vendor_name"] for v in vendors]
        assert "Test Vendor Ltd" in names
        assert "No PAN Vendor" in names
    
    def test_get_vendor_returns_none_for_unknown_id(self, erp_client):
        """get_vendor should return None for unknown ID."""
        assert erp_client.get_vendor("VND-UNKNOWN") is None


class TestResponseStructure:
    """Tests for ERPResponse structure."""
    
    def test_success_response_has_all_fields(self, erp_client, valid_vendor):
        """Success response should have all required fields."""
        response = erp_client.create_vendor(valid_vendor)
        
        assert hasattr(response, 'success')
        assert hasattr(response, 'error_code')
        assert hasattr(response, 'message')
        assert hasattr(response, 'vendor_id')
        assert hasattr(response, 'erp_reference')
    
    def test_error_response_has_no_vendor_id(self, erp_client):
        """Error response should not have vendor_id."""
        vendor = Vendor(vendor_name="")  # Invalid
        
        response = erp_client.create_vendor(vendor)
        
        assert response.success is False
        assert response.vendor_id is None
        assert response.erp_reference is None


# Run Tests
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
