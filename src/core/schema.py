from pydantic import BaseModel, Field
from typing import Optional, Dict, Literal


class ExtractedField(BaseModel):
    value: Optional[str] = Field(
        default=None,
        description="Extracted value"
    )
    confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Confidence score between 0.0 and 1.0"
    )


class ExtractionResult(BaseModel):
    model_config = {"exclude_none": True}
    
    document_type: Literal["PAN", "GST", "CHEQUE", "UNKNOWN"] = Field(
        ...,
        description="Detected document type"
    )
    status: Literal["success", "failure"] = Field(
        ...,
        description="Extraction status"
    )
    message: Optional[str] = Field(
        default=None,
        description="Error message or status details"
    )
    fields: Dict[str, ExtractedField] = Field(
        default_factory=dict,
        description="Dynamically extracted fields"
    )
