"""
schemas.py
----------
Pydantic request and response models for the FastAPI service.
These define the exact JSON shape exchanged with the Java backend.
"""

from pydantic import BaseModel, Field


class FraudCheckRequest(BaseModel):
    """
    The Java backend sends this when submitting a deferment document for checking.
    The document text should be pre-extracted before calling this endpoint.
    """
    application_id: str = Field(..., description="Unique deferment application ID")
    document_text: str = Field(..., description="Extracted text from the supporting document")
    document_type: str = Field(
        default="enrollment_letter",
        description="Type of document: enrollment_letter | employment_letter | travel_doc | medical_cert"
    )


class FraudCheckResponse(BaseModel):
    """
    Returned to the Java backend after classification.
    """
    application_id: str
    label: str = Field(..., description="'genuine' or 'fraudulent'")
    fraud_score: float = Field(..., description="Confidence that the document is fraudulent (0.0 - 1.0)")
    genuine_score: float = Field(..., description="Confidence that the document is genuine (0.0 - 1.0)")
    recommendation: str = Field(..., description="'APPROVE', 'FLAG_FOR_REVIEW', or 'REJECT'")
    flags: list[str] = Field(default_factory=list, description="List of specific issues detected")


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
