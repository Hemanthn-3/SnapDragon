"""
NEXUS Phase 7: Evidence Verification Layer Schemas
Strict structured models linking every important finding to traceable local evidence.
"""

from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, field_validator


class VerificationStatus(str, Enum):
    """
    Standard verification classifications for extracted claims and findings.
    """
    VERIFIED = "VERIFIED"
    SUPPORTED = "SUPPORTED"
    UNCERTAIN = "UNCERTAIN"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"


class EvidenceItem(BaseModel):
    """
    Traceable reference to local document evidence.
    """
    document: str = Field(..., description="Document filename or identifier (e.g. 'Safety_Manual.pdf')")
    page: Optional[int] = Field(default=None, ge=1, description="Page number where evidence appears")
    source_chunk: str = Field(..., description="Unique chunk identifier or location reference")
    relevant_text: str = Field(..., min_length=1, description="Direct excerpt or relevant text chunk")

    @field_validator("document", "source_chunk", "relevant_text")
    @classmethod
    def strip_whitespace(cls, v):
        if isinstance(v, str):
            return v.strip()
        return v


class Finding(BaseModel):
    """
    Structured representation of an assertion or finding grounded by evidence.
    """
    id: str = Field(..., description="Unique identifier for finding (e.g. 'finding_1')")
    claim: str = Field(..., min_length=1, description="The substantive factual statement being evaluated")
    evidence: List[EvidenceItem] = Field(default_factory=list, description="All supporting or reference evidence items")
    confidence: float = Field(default=0.0, ge=0.0, le=1.0, description="Verification confidence score (0.0 to 1.0)")
    verification_status: VerificationStatus = Field(
        default=VerificationStatus.INSUFFICIENT_EVIDENCE,
        description="Assigned verification status",
    )
    explanation: str = Field(
        ...,
        description="Concise evidence explanation. Strictly excludes hidden chain-of-thought reasoning.",
    )

    @field_validator("explanation")
    @classmethod
    def validate_concise_explanation(cls, v: str) -> str:
        # Guarantee no chain of thought leak (e.g. "<think>", "Let's think step by step", scratchpad tokens)
        banned_phrases = ["<think>", "</think>", "let's think step by step", "scratchpad", "chain of thought:"]
        lower = v.lower()
        for banned in banned_phrases:
            if banned in lower:
                raise ValueError(f"Exposed chain-of-thought token detected in explanation: '{banned}'")
        return v.strip()


class VerificationRequest(BaseModel):
    """
    Request to verify an assertion against provided or retrieved evidence.
    """
    claim: str = Field(..., min_length=1, description="Claim to verify")
    finding_id: Optional[str] = Field(default=None, description="Optional finding ID")
    evidence: Optional[List[EvidenceItem]] = Field(default=None, description="Explicit evidence chunks to evaluate")
    auto_retrieve: bool = Field(default=True, description="Whether to query local vector store if evidence omitted")
    top_k: int = Field(default=5, ge=1, le=10, description="Max citations to retrieve from knowledge store")


class BatchVerificationRequest(BaseModel):
    """
    Request to verify multiple claims in batch.
    """
    claims: List[str] = Field(..., min_length=1, description="List of claims to verify")
    auto_retrieve: bool = Field(default=True)


class VerificationResponse(BaseModel):
    """
    Result of an evidence verification process.
    """
    finding: Finding = Field(..., description="Grounded finding with assigned status and evidence")
    retrieved_citation_count: int = Field(default=0, description="Number of source citations analyzed")
