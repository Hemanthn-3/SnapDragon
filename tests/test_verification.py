"""
NEXUS Phase 7: Evidence Verification Layer Tests
Tests all verification criteria:
1. Claim is supported (single-source SUPPORTED, multi-source VERIFIED)
2. Claim is contradicted (strictly NEVER VERIFIED)
3. Evidence is missing (strictly INSUFFICIENT_EVIDENCE)
4. Unrelated evidence rejection
5. Zero hidden chain-of-thought leaks
6. REST API integration (/verification/verify and /verification/batch)
"""

import pytest
from pydantic import ValidationError

from backend.verification.schemas import (
    Finding,
    EvidenceItem,
    VerificationStatus,
    VerificationRequest,
    BatchVerificationRequest,
)
from backend.verification.verifier import EvidenceVerifier, evidence_verifier


# =====================================================================
# 1. Test Claim is Supported (Single Source vs Multi-Source)
# =====================================================================

def test_claim_is_supported_single_source():
    verifier = EvidenceVerifier()
    claim = "Turbine bearing vibration threshold is 8.0 mm/s."
    evidence = [
        EvidenceItem(
            document="Safety_Manual.pdf",
            page=17,
            source_chunk="chunk_safety_017",
            relevant_text="Maximum allowable bearing vibration threshold is 8.0 mm/s RMS under operational load.",
        )
    ]
    req = VerificationRequest(claim=claim, evidence=evidence, auto_retrieve=False)
    res = verifier.verify_claim(req)

    finding = res.finding
    assert finding.verification_status == VerificationStatus.SUPPORTED
    assert 0.70 <= finding.confidence <= 0.85
    assert len(finding.evidence) == 1
    assert "Safety_Manual.pdf" in finding.explanation
    assert "Page 17" in finding.explanation
    # Ensure no chain of thought leak
    assert "<think>" not in finding.explanation
    assert "scratchpad" not in finding.explanation.lower()


def test_claim_is_verified_multi_source():
    """
    User prompt example:
    Finding: "Inspection value exceeds the reference threshold."
    Evidence:
      Safety_Manual.pdf — Page 17
      Inspection_Report.pdf — Page 4
    Status: VERIFIED
    """
    verifier = EvidenceVerifier()
    claim = "Inspection value exceeds the reference threshold."
    evidence = [
        EvidenceItem(
            document="Safety_Manual.pdf",
            page=17,
            source_chunk="chunk_safety_017",
            relevant_text="Operational standard: Maximum reference threshold for turbine vibration is 8.0 mm/s.",
        ),
        EvidenceItem(
            document="Inspection_Report.pdf",
            page=4,
            source_chunk="chunk_insp_004",
            relevant_text="Field measurement: Turbine bearing vibration reached 14.2 mm/s, which exceeds the reference threshold.",
        ),
    ]

    req = VerificationRequest(claim=claim, evidence=evidence, auto_retrieve=False)
    res = verifier.verify_claim(req)

    finding = res.finding
    assert finding.verification_status == VerificationStatus.VERIFIED
    assert finding.confidence >= 0.88
    assert len(finding.evidence) == 2
    assert "Safety_Manual.pdf — Page 17" in finding.explanation
    assert "Inspection_Report.pdf — Page 4" in finding.explanation
    # Must never expose internal chain of thought
    assert "<think>" not in finding.explanation
    assert "let's think step by step" not in finding.explanation.lower()


# =====================================================================
# 2. Test Claim is Contradicted (Never VERIFIED)
# =====================================================================

def test_claim_is_contradicted_by_evidence():
    """
    CRITICAL RULE: If evidence cannot support a claim or contradicts it:
    Do NOT mark it VERIFIED.
    """
    verifier = EvidenceVerifier()
    claim = "Inspection value exceeds the reference threshold."
    evidence = [
        EvidenceItem(
            document="Inspection_Report.pdf",
            page=4,
            source_chunk="chunk_insp_004",
            relevant_text="All vibration inspection values are strictly within threshold and compliant with normal limits.",
        )
    ]

    req = VerificationRequest(claim=claim, evidence=evidence, auto_retrieve=False)
    res = verifier.verify_claim(req)

    finding = res.finding
    # Must NEVER be VERIFIED
    assert finding.verification_status != VerificationStatus.VERIFIED
    assert finding.verification_status != VerificationStatus.SUPPORTED
    assert finding.verification_status == VerificationStatus.UNCERTAIN
    assert finding.confidence <= 0.25
    assert "Contradiction identified" in finding.explanation
    assert "within threshold" in finding.explanation


def test_compliance_claim_contradicted_by_violation():
    verifier = EvidenceVerifier()
    claim = "Turbine vibration remained normal and within limits."
    evidence = [
        EvidenceItem(
            document="Telemetry_Log.pdf",
            page=12,
            source_chunk="chunk_telemetry_012",
            relevant_text="Alarm triggered: Turbine bearing 1 exceeds limit with critical deviation of 16.4 mm/s.",
        )
    ]

    req = VerificationRequest(claim=claim, evidence=evidence, auto_retrieve=False)
    res = verifier.verify_claim(req)

    finding = res.finding
    assert finding.verification_status != VerificationStatus.VERIFIED
    assert finding.verification_status == VerificationStatus.UNCERTAIN
    assert finding.confidence <= 0.25


# =====================================================================
# 3. Test Evidence is Missing (Never VERIFIED)
# =====================================================================

def test_claim_with_missing_evidence():
    verifier = EvidenceVerifier()
    claim = "The emergency generator was upgraded to a 500kW diesel engine in 2023."
    # Explicitly empty evidence, auto_retrieve=False
    req = VerificationRequest(claim=claim, evidence=[], auto_retrieve=False)
    res = verifier.verify_claim(req)

    finding = res.finding
    # Must NEVER be VERIFIED
    assert finding.verification_status != VerificationStatus.VERIFIED
    assert finding.verification_status != VerificationStatus.SUPPORTED
    assert finding.verification_status == VerificationStatus.INSUFFICIENT_EVIDENCE
    assert finding.confidence == 0.0
    assert "No evidence available" in finding.explanation


def test_unrelated_evidence_rejected_as_insufficient():
    verifier = EvidenceVerifier()
    claim = "Inspection value exceeds the reference threshold."
    evidence = [
        EvidenceItem(
            document="Cafeteria_Schedule.txt",
            page=1,
            source_chunk="chunk_cafe_001",
            relevant_text="The cafeteria serves hot breakfast starting at 7:00 AM on weekdays.",
        )
    ]

    req = VerificationRequest(claim=claim, evidence=evidence, auto_retrieve=False)
    res = verifier.verify_claim(req)

    finding = res.finding
    assert finding.verification_status != VerificationStatus.VERIFIED
    assert finding.verification_status == VerificationStatus.INSUFFICIENT_EVIDENCE
    assert finding.confidence <= 0.20
    assert "does not sufficiently substantiate" in finding.explanation or "does not contain facts" in finding.explanation


# =====================================================================
# 4. Test Chain-of-Thought Protection
# =====================================================================

def test_chain_of_thought_leak_rejected_by_schema():
    """Verifies that attempting to inject chain-of-thought into Finding is strictly rejected."""
    with pytest.raises(ValidationError) as exc_info:
        Finding(
            id="finding_leak_test",
            claim="Test claim",
            evidence=[],
            confidence=0.5,
            verification_status=VerificationStatus.SUPPORTED,
            explanation="<think>Let's consider whether this claim is true step by step</think> Evidence: Doc1",
        )
    assert "Exposed chain-of-thought token detected" in str(exc_info.value)


# =====================================================================
# 5. Test REST API Endpoints
# =====================================================================

def test_api_verify_claim_endpoint(client):
    payload = {
        "claim": "Bearing vibration limit is 8.0 mm/s.",
        "evidence": [
            {
                "document": "Safety_Specs.pdf",
                "page": 5,
                "source_chunk": "chunk_005",
                "relevant_text": "Bearing vibration limit is 8.0 mm/s under continuous operation.",
            }
        ],
        "auto_retrieve": False,
    }
    response = client.post("/verification/verify", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "finding" in data
    assert data["finding"]["verification_status"] == "SUPPORTED"
    assert data["finding"]["confidence"] >= 0.70
    assert "Safety_Specs.pdf — Page 5" in data["finding"]["explanation"]


def test_api_verify_contradicted_claim(client):
    payload = {
        "claim": "Inspection value exceeds reference threshold.",
        "evidence": [
            {
                "document": "Report.pdf",
                "page": 1,
                "source_chunk": "chunk_001",
                "relevant_text": "Values are completely normal and within threshold.",
            }
        ],
        "auto_retrieve": False,
    }
    response = client.post("/verification/verify", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["finding"]["verification_status"] != "VERIFIED"
    assert data["finding"]["verification_status"] == "UNCERTAIN"


def test_api_verify_missing_evidence(client):
    payload = {
        "claim": "Completely unsubstantiated assertion without any documents.",
        "evidence": [],
        "auto_retrieve": False,
    }
    response = client.post("/verification/verify", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["finding"]["verification_status"] == "INSUFFICIENT_EVIDENCE"
    assert data["finding"]["confidence"] == 0.0


def test_api_verify_empty_claim_rejected(client):
    response = client.post("/verification/verify", json={"claim": "   "})
    assert response.status_code == 422


def test_api_batch_verification(client):
    payload = {
        "claims": [
            "Turbine vibration exceeds limit.",
            "Normal operating temperature is 45C.",
        ],
        "auto_retrieve": False,
    }
    response = client.post("/verification/batch", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) == 2
    for item in data:
        assert "verification_status" in item
        assert "confidence" in item
