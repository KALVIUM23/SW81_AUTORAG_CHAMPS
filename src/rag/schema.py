"""
Pydantic Schemas for Structured Diagnostic Responses (Topic 3.17)
"""
from typing import List, Optional
from pydantic import BaseModel, Field


class Citation(BaseModel):
    document_id: str = Field(description="Unique document identifier")
    document_title: str = Field(description="Title of the manual or TSB")
    version: str = Field(description="Document version or release tag")
    section: str = Field(description="Section heading or number")
    page_number: int = Field(description="Page number in manual")
    relevance_snippet: str = Field(description="Exact excerpt verifying the instruction")


class DiagnosticStep(BaseModel):
    step_number: int = Field(description="Sequential step index")
    title: str = Field(description="Concise step title (e.g., 'Exhaust Leak Inspection')")
    procedure: str = Field(description="Step-by-step technical instruction")
    tools_required: Optional[List[str]] = Field(default=[], description="Specialized workshop tools")
    torque_or_electrical_spec: Optional[str] = Field(default=None, description="Torque specs or voltage thresholds")
    citation_id: Optional[str] = Field(default=None, description="Linked citation document_id")


class SafetyAdvisory(BaseModel):
    has_active_recall: bool = Field(default=False)
    recall_id: Optional[str] = Field(default=None)
    severity: Optional[str] = Field(default="LOW", description="LOW, MEDIUM, CRITICAL")
    warning_message: Optional[str] = Field(default=None)


class DiagnosticResponse(BaseModel):
    vehicle_context: str = Field(description="Vehicle summary (e.g., '2025 Model X Hybrid [India]')")
    dtc_code: Optional[str] = Field(default=None, description="Diagnostic trouble code")
    safety_advisory: SafetyAdvisory
    summary: str = Field(description="Executive diagnosis summary")
    steps: List[DiagnosticStep] = Field(default=[], description="Actionable repair steps")
    citations: List[Citation] = Field(default=[], description="Attributed manual citations")
    is_refusal: bool = Field(default=False, description="True if prompt refuses due to missing docs")
    refusal_reason: Optional[str] = Field(default=None)