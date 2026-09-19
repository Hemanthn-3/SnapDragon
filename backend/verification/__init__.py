"""
NEXUS Verification Package
Evidence verification layer linking findings to traceable local document chunks.
"""

from backend.verification.schemas import (
    VerificationStatus,
    EvidenceItem,
    Finding,
    VerificationRequest,
    BatchVerificationRequest,
    VerificationResponse,
)
from backend.verification.verifier import EvidenceVerifier, evidence_verifier

__all__ = [
    "VerificationStatus",
    "EvidenceItem",
    "Finding",
    "VerificationRequest",
    "BatchVerificationRequest",
    "VerificationResponse",
    "EvidenceVerifier",
    "evidence_verifier",
]
