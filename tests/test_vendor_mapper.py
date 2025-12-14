"""
Unit Tests for VendorMapper

Tests the transformation layer that converts OCR extraction
output to canonical Vendor model.
"""
import pytest
from datetime import datetime

from src.integration.vendor_mapper import VendorMapper
from src.core.vendor_model import Vendor, BankDetails


@pytest.fixture
def mapper():
    """Create a VendorMapper instance."""
    return VendorMapper()


@pytest.fixture
def pan_extraction_result():
    """Sample successful PAN extraction result."""
    return {
        "document_type": "PAN",
        "status": "success",
        "fields": {
            "pan_number": {"value": "ABCDE1234F", "confidence": 0.98},
            "name": {"value": "JOHN DOE", "confidence": 0.95},
            "date_of_birth": {"value": "15/08/1990", "confidence": 0.90}
        }
    }


@pytest.fixture
def gst_extraction_result():
    """Sample successful GST extraction result."""
    return {
        "document_type": "GST",
        "status": "success",
        "fields": {
            "gstin": {"value": "27ABCDE1234F1Z5", "confidence": 0.97},
            "legal_name": {"value": "ACME CORPORATION PVT LTD", "confidence": 0.92},
            "address": {"value": "123 MAIN STREET, MUMBAI 400001", "confidence": 0.85},
            "registration_date": {"value": "01/04/2020", "confidence": 0.90}
        }
    }


@pytest.fixture
def bank_extraction_result():
    """Sample successful bank document extraction result."""
    return {
        "document_type": "CHEQUE",
        "status": "success",
        "fields": {
            "payee_name": {"value": "JANE SMITH", "confidence": 0.88},
            "account_number": {"value": "123456789012", "confidence": 0.95},
            "ifsc_code": {"value": "HDFC0001234", "confidence": 0.93}
        }
    }


@pytest.fixture
def failed_extraction_result():
    """Sample failed extraction result."""
    return {
        "document_type": "UNKNOWN",
        "status": "failure",
        "message": "Could not read document",
        "fields": {}
    }


class TestBasicMapping:
    """Tests for basic mapping functionality."""
    
    def test_maps_pan_to_vendor(self, mapper, pan_extraction_result):
        """Should successfully map PAN extraction to Vendor."""
        vendor = mapper.map_from_extraction(pan_extraction_result)
        
        assert vendor is not None
        assert vendor.vendor_name == "JOHN DOE"
        assert vendor.pan == "ABCDE1234F"
        assert vendor.source_document_type == "PAN"
    
    def test_maps_gst_to_vendor(self, mapper, gst_extraction_result):
        """Should successfully map GST extraction to Vendor."""
        vendor = mapper.map_from_extraction(gst_extraction_result)
        
        assert vendor is not None
        assert vendor.vendor_name == "ACME CORPORATION PVT LTD"
        assert vendor.gstin == "27ABCDE1234F1Z5"
        assert vendor.address == "123 MAIN STREET, MUMBAI 400001"
        assert vendor.source_document_type == "GST"
    
    def test_maps_bank_to_vendor(self, mapper, bank_extraction_result):
        """Should successfully map bank extraction to Vendor."""
        vendor = mapper.map_from_extraction(bank_extraction_result)
        
        assert vendor is not None
        assert vendor.vendor_name == "JANE SMITH"
        assert vendor.bank_details is not None
        assert vendor.bank_details.account_number == "123456789012"
        assert vendor.bank_details.ifsc_code == "HDFC0001234"
        assert vendor.source_document_type == "CHEQUE"
    
    def test_failed_extraction_returns_none(self, mapper, failed_extraction_result):
        """Should return None for failed extractions."""
        vendor = mapper.map_from_extraction(failed_extraction_result)
        assert vendor is None

class TestConfidenceScore:
    """Tests for aggregate confidence score calculation."""
    
    def test_confidence_score_is_calculated(self, mapper, pan_extraction_result):
        """Vendor should have an aggregate confidence score."""
        vendor = mapper.map_from_extraction(pan_extraction_result)
        
        assert vendor is not None
        assert 0.0 <= vendor.confidence_score <= 1.0
    
    def test_higher_confidence_fields_yield_higher_score(self, mapper):
        """High confidence extraction should yield high aggregate score."""
        high_confidence_result = {
            "document_type": "PAN",
            "status": "success",
            "fields": {
                "pan_number": {"value": "ABCDE1234F", "confidence": 0.99},
                "name": {"value": "JOHN DOE", "confidence": 0.99}
            }
        }
        
        vendor = mapper.map_from_extraction(high_confidence_result)
        assert vendor.confidence_score > 0.9

class TestMissingFields:
    """Tests for graceful handling of missing fields."""
    
    def test_handles_missing_pan(self, mapper):
        """Should map vendor without PAN number."""
        result = {
            "document_type": "GST",
            "status": "success",
            "fields": {
                "legal_name": {"value": "TEST COMPANY", "confidence": 0.90}
            }
        }
        
        vendor = mapper.map_from_extraction(result)
        assert vendor is not None
        assert vendor.vendor_name == "TEST COMPANY"
        assert vendor.pan is None
    
    def test_handles_missing_bank_details(self, mapper, pan_extraction_result):
        """PAN extraction should have no bank details."""
        vendor = mapper.map_from_extraction(pan_extraction_result)
        
        assert vendor is not None
        assert vendor.bank_details is None
    
    def test_returns_none_without_name(self, mapper):
        """Should return None if no name can be extracted."""
        result = {
            "document_type": "PAN",
            "status": "success",
            "fields": {
                "pan_number": {"value": "ABCDE1234F", "confidence": 0.98}
                # No name field
            }
        }
        
        vendor = mapper.map_from_extraction(result)
        assert vendor is None
    
    def test_handles_empty_fields(self, mapper):
        """Should handle empty fields dictionary."""
        result = {
            "document_type": "UNKNOWN",
            "status": "success",
            "fields": {}
        }
        
        vendor = mapper.map_from_extraction(result)
        assert vendor is None

class TestDataNormalization:
    """Tests for data normalization during mapping."""
    
    def test_normalizes_pan_to_uppercase(self, mapper):
        """PAN should be normalized to uppercase."""
        result = {
            "document_type": "PAN",
            "status": "success",
            "fields": {
                "pan_number": {"value": "abcde1234f", "confidence": 0.98},
                "name": {"value": "John Doe", "confidence": 0.95}
            }
        }
        
        vendor = mapper.map_from_extraction(result)
        assert vendor.pan == "ABCDE1234F"
    
    def test_normalizes_gstin_to_uppercase(self, mapper):
        """GSTIN should be normalized to uppercase."""
        result = {
            "document_type": "GST",
            "status": "success",
            "fields": {
                "gstin": {"value": "27abcde1234f1z5", "confidence": 0.97},
                "legal_name": {"value": "Test Company", "confidence": 0.92}
            }
        }
        
        vendor = mapper.map_from_extraction(result)
        assert vendor.gstin == "27ABCDE1234F1Z5"
    
    def test_normalizes_ifsc_to_uppercase(self, mapper):
        """IFSC should be normalized to uppercase."""
        result = {
            "document_type": "CHEQUE",
            "status": "success",
            "fields": {
                "payee_name": {"value": "John Doe", "confidence": 0.88},
                "ifsc_code": {"value": "hdfc0001234", "confidence": 0.93}
            }
        }
        
        vendor = mapper.map_from_extraction(result)
        assert vendor.bank_details.ifsc_code == "HDFC0001234"


class TestBatchMapping:
    """Tests for batch mapping functionality."""
    
    def test_map_multiple_results(self, mapper, pan_extraction_result, gst_extraction_result):
        """Should map multiple extraction results."""
        results = [pan_extraction_result, gst_extraction_result]
        
        vendors = mapper.map_multiple(results)
        
        assert len(vendors) == 2
        assert vendors[0].vendor_name == "JOHN DOE"
        assert vendors[1].vendor_name == "ACME CORPORATION PVT LTD"
    
    def test_filters_failed_extractions(self, mapper, pan_extraction_result, failed_extraction_result):
        """Should filter out failed extractions from batch."""
        results = [pan_extraction_result, failed_extraction_result]
        
        vendors = mapper.map_multiple(results)
        
        assert len(vendors) == 1
        assert vendors[0].vendor_name == "JOHN DOE"
    
    def test_empty_list_returns_empty(self, mapper):
        """Empty input should return empty list."""
        vendors = mapper.map_multiple([])
        assert vendors == []


class TestVendorValidation:
    """Tests for Vendor model validation."""
    
    def test_valid_vendor_passes_validation(self, mapper, pan_extraction_result):
        """Valid vendor should pass validation."""
        vendor = mapper.map_from_extraction(pan_extraction_result)
        
        is_valid, errors = vendor.validate()
        assert is_valid is True
        assert len(errors) == 0
    
    def test_invalid_pan_fails_validation(self):
        """Vendor with invalid PAN should fail validation."""
        vendor = Vendor(
            vendor_name="Test",
            pan="INVALID"  # Wrong format
        )
        
        is_valid, errors = vendor.validate()
        assert is_valid is False
        assert any("PAN" in e for e in errors)
    
    def test_invalid_gstin_fails_validation(self):
        """Vendor with invalid GSTIN should fail validation."""
        vendor = Vendor(
            vendor_name="Test",
            gstin="INVALID"  # Wrong format
        )
        
        is_valid, errors = vendor.validate()
        assert is_valid is False
        assert any("GSTIN" in e for e in errors)


# Run Tests
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
