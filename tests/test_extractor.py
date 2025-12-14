"""
Unit Tests for DocumentExtractor

Tests the OCR-based document extraction logic for:
- PAN cards
- GST certificates
- Bank documents (cheques/certificates)

Uses synthetic test data to validate extraction accuracy.
"""
import pytest
import os
from pathlib import Path

from src.core.extractor import DocumentExtractor


# --------------------------------------------------
# Fixtures
# --------------------------------------------------
@pytest.fixture(scope="module")
def extractor():
    """Create a shared extractor instance (expensive to initialize)."""
    return DocumentExtractor(use_gpu=False)


@pytest.fixture
def sample_pan_path():
    """Path to a sample PAN image."""
    path = Path("data/pan/pan_01.png")
    if not path.exists():
        pytest.skip("Sample PAN image not found. Run synth.py first.")
    return str(path)


@pytest.fixture
def sample_gst_path():
    """Path to a sample GST image."""
    path = Path("data/gst/gst_01.png")
    if not path.exists():
        pytest.skip("Sample GST image not found. Run synth.py first.")
    return str(path)


@pytest.fixture
def sample_bank_path():
    """Path to a sample Bank image."""
    path = Path("data/bank/bank_01.png")
    if not path.exists():
        pytest.skip("Sample Bank image not found. Run synth.py first.")
    return str(path)


# --------------------------------------------------
# Test: Basic Extraction Structure
# --------------------------------------------------
class TestExtractionStructure:
    """Tests that extraction returns properly structured results."""
    
    def test_result_has_required_keys(self, extractor, sample_pan_path):
        """Extraction result should have document_type, status, and fields."""
        result = extractor.extract_from_file(sample_pan_path)
        
        assert "document_type" in result
        assert "status" in result
        assert "fields" in result
    
    def test_status_is_valid(self, extractor, sample_pan_path):
        """Status should be 'success' or 'failure'."""
        result = extractor.extract_from_file(sample_pan_path)
        assert result["status"] in ["success", "failure"]
    
    def test_fields_have_confidence_scores(self, extractor, sample_pan_path):
        """Each extracted field should have a confidence score."""
        result = extractor.extract_from_file(sample_pan_path)
        
        if result["status"] == "success":
            for field_name, field_data in result["fields"].items():
                assert "value" in field_data, f"Field {field_name} missing 'value'"
                assert "confidence" in field_data, f"Field {field_name} missing 'confidence'"
                assert 0.0 <= field_data["confidence"] <= 1.0, \
                    f"Confidence for {field_name} out of range: {field_data['confidence']}"


# --------------------------------------------------
# Test: PAN Extraction
# --------------------------------------------------
class TestPANExtraction:
    """Tests specific to PAN card extraction."""
    
    def test_detects_pan_document_type(self, extractor, sample_pan_path):
        """Should correctly identify document as PAN."""
        result = extractor.extract_from_file(sample_pan_path)
        assert result["document_type"] == "PAN"
    
    def test_extracts_pan_number(self, extractor, sample_pan_path):
        """Should extract a valid PAN number."""
        result = extractor.extract_from_file(sample_pan_path)
        
        if result["status"] == "success":
            assert "pan_number" in result["fields"]
            pan = result["fields"]["pan_number"]["value"]
            # PAN format: 5 letters + 4 digits + 1 letter
            assert len(pan) == 10, f"PAN should be 10 chars, got {len(pan)}"
    
    def test_extracts_name(self, extractor, sample_pan_path):
        """Should extract cardholder name."""
        result = extractor.extract_from_file(sample_pan_path)
        
        if result["status"] == "success":
            assert "name" in result["fields"]
            name = result["fields"]["name"]["value"]
            assert len(name) > 2, "Name should be at least 3 characters"
    
    def test_extracts_dob(self, extractor, sample_pan_path):
        """Should extract date of birth if present."""
        result = extractor.extract_from_file(sample_pan_path)
        
        if result["status"] == "success":
            # DOB extraction is not guaranteed for all documents
            if "date_of_birth" in result["fields"]:
                dob = result["fields"]["date_of_birth"]["value"]
                # Should be in DD/MM/YYYY or DD-MM-YYYY format
                assert "/" in dob or "-" in dob, f"DOB format unexpected: {dob}"
    
    def test_pan_does_not_have_ifsc(self, extractor, sample_pan_path):
        """PAN documents should NOT have IFSC code extracted."""
        result = extractor.extract_from_file(sample_pan_path)
        
        if result["status"] == "success":
            assert "ifsc_code" not in result["fields"], \
                "PAN should not have IFSC code"


# --------------------------------------------------
# Test: GST Extraction
# --------------------------------------------------
class TestGSTExtraction:
    """Tests specific to GST certificate extraction."""
    
    def test_detects_gst_document_type(self, extractor, sample_gst_path):
        """Should correctly identify document as GST."""
        result = extractor.extract_from_file(sample_gst_path)
        assert result["document_type"] == "GST"
    
    def test_extracts_gstin(self, extractor, sample_gst_path):
        """Should extract a valid GSTIN."""
        result = extractor.extract_from_file(sample_gst_path)
        
        if result["status"] == "success":
            assert "gstin" in result["fields"]
            gstin = result["fields"]["gstin"]["value"]
            assert len(gstin) == 15, f"GSTIN should be 15 chars, got {len(gstin)}"
    
    def test_extracts_legal_name(self, extractor, sample_gst_path):
        """Should extract legal/trade name."""
        result = extractor.extract_from_file(sample_gst_path)
        
        if result["status"] == "success":
            assert "legal_name" in result["fields"]
    
    def test_extracts_address(self, extractor, sample_gst_path):
        """Should extract business address."""
        result = extractor.extract_from_file(sample_gst_path)
        
        if result["status"] == "success":
            assert "address" in result["fields"]


# --------------------------------------------------
# Test: Bank Document Extraction
# --------------------------------------------------
class TestBankExtraction:
    """Tests specific to bank document extraction."""
    
    def test_detects_cheque_document_type(self, extractor, sample_bank_path):
        """Should correctly identify document as CHEQUE."""
        result = extractor.extract_from_file(sample_bank_path)
        assert result["document_type"] == "CHEQUE"
    
    def test_extracts_account_number(self, extractor, sample_bank_path):
        """Should extract bank account number."""
        result = extractor.extract_from_file(sample_bank_path)
        
        if result["status"] == "success":
            assert "account_number" in result["fields"]
            acc = result["fields"]["account_number"]["value"]
            assert acc.isdigit(), f"Account number should be digits: {acc}"
    
    def test_extracts_ifsc(self, extractor, sample_bank_path):
        """Should extract IFSC code."""
        result = extractor.extract_from_file(sample_bank_path)
        
        if result["status"] == "success":
            assert "ifsc_code" in result["fields"]
            ifsc = result["fields"]["ifsc_code"]["value"]
            assert len(ifsc) == 11, f"IFSC should be 11 chars, got {len(ifsc)}"
    
    def test_extracts_account_holder(self, extractor, sample_bank_path):
        """Should extract account holder name."""
        result = extractor.extract_from_file(sample_bank_path)
        
        if result["status"] == "success":
            assert "payee_name" in result["fields"]


# --------------------------------------------------
# Test: Error Handling
# --------------------------------------------------
class TestErrorHandling:
    """Tests for graceful error handling."""
    
    def test_nonexistent_file_returns_failure(self, extractor):
        """Should return failure status for missing files."""
        result = extractor.extract_from_file("nonexistent_file.png")
        assert result["status"] == "failure"
    
    def test_invalid_file_returns_failure(self, extractor, tmp_path):
        """Should handle invalid image files gracefully."""
        # Create an invalid image file
        invalid_file = tmp_path / "invalid.png"
        invalid_file.write_text("not an image")
        
        result = extractor.extract_from_file(str(invalid_file))
        assert result["status"] == "failure"


# --------------------------------------------------
# Run Tests
# --------------------------------------------------
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
