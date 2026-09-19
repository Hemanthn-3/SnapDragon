"""
NEXUS Phase 7: Evidence Verification Route
Endpoints for grounding and verifying claims against local document evidence.
"""

from typing import List
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from backend.database import get_db
from backend.logger import get_logger
from backend.verification.schemas import (
    VerificationRequest,
    BatchVerificationRequest,
    VerificationResponse,
    Finding,
)
from backend.verification.verifier import evidence_verifier

logger = get_logger("nexus.routes_verification")

router = APIRouter(prefix="/verification", tags=["Evidence Verification"])


@router.post(
    "/verify",
    response_model=VerificationResponse,
    summary="Verify Factual Claim",
    description="Evaluates a claim against explicit or retrieved local evidence. Returns assigned status and concise explanation.",
)
def verify_claim(request: VerificationRequest, db: Session = Depends(get_db)):
    if not request.claim.strip():
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Claim cannot be empty or whitespace.",
        )
    logger.info(f"[VERIFY] Verifying claim: '{request.claim[:80]}...'")
    response = evidence_verifier.verify_claim(request, db=db)
    return response


@router.post(
    "/batch",
    response_model=List[Finding],
    summary="Batch Verify Factual Claims",
    description="Sequentially verifies a collection of claims, returning structured findings.",
)
def verify_batch_claims(request: BatchVerificationRequest):
    if not request.claims:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Claims list cannot be empty.",
        )
    logger.info(f"[VERIFY] Batch verifying {len(request.claims)} claims.")
    findings = evidence_verifier.verify_batch(request.claims, auto_retrieve=request.auto_retrieve)
    return findings
