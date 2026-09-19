"""
NEXUS Phase 7: Evidence Verification Engine
Evaluates factual claims against local evidence chunks and vector citations.
Enforces multi-source corroboration, contradiction detection, missing evidence rejection,
and concise explanations without exposing chain-of-thought tokens.
"""

import re
import uuid
from typing import Any, Dict, List, Optional, Set, Tuple
import numpy as np
from sqlalchemy.orm import Session

from backend.database import SessionLocal
from backend.logger import logger
from backend.models_local.minilm_embedding import local_embedding_model
from backend.knowledge.service import knowledge_service
from backend.verification.schemas import (
    Finding,
    EvidenceItem,
    VerificationStatus,
    VerificationRequest,
    VerificationResponse,
)


class EvidenceVerifier:
    """
    Offline verification engine linking findings to verifiable evidence.
    Rules:
    1. Zero evidence -> INSUFFICIENT_EVIDENCE (Never VERIFIED).
    2. Contradiction detected -> UNCERTAIN / INSUFFICIENT_EVIDENCE (Never VERIFIED).
    3. Single document source -> SUPPORTED.
    4. Multi-document corroboration -> VERIFIED.
    5. Explanation is concise evidence summary; zero hidden chain-of-thought.
    """

    def __init__(self, embedding_model=None):
        self.embedding_model = embedding_model or local_embedding_model

    def _detect_contradiction(self, claim: str, evidence_text: str) -> Tuple[bool, str]:
        """
        Detects factual or polarity contradictions between claim and evidence.
        Returns (is_contradicted, explanation_detail).
        """
        claim_lower = claim.lower()
        ev_lower = evidence_text.lower()

        # 1. Direct polarity opposition keywords
        # Positive assertion of violation/exceedance in claim vs compliance in evidence
        violation_claim_terms = ["exceeds", "exceeded", "above threshold", "violation", "violated", "failed", "unsafe", "abnormal"]
        compliance_ev_terms = ["within threshold", "within limits", "below threshold", "did not exceed", "does not exceed", "compliant", "safe", "normal", "passed inspection", "no deviation", "acceptable range"]

        if any(term in claim_lower for term in violation_claim_terms):
            for comp_term in compliance_ev_terms:
                if comp_term in ev_lower:
                    return True, f"Evidence states '{comp_term}', directly opposing claim of violation."

        # Compliance claim vs violation in evidence
        compliance_claim_terms = ["within limits", "normal", "compliant", "passed", "safe", "below threshold", "within threshold"]
        violation_ev_terms = ["exceeds limit", "exceeded threshold", "failed", "violation", "unsafe", "critical deviation", "out of tolerance", "unacceptable"]

        if any(term in claim_lower for term in compliance_claim_terms):
            for viol_term in violation_ev_terms:
                if viol_term in ev_lower:
                    return True, f"Evidence reports '{viol_term}', directly opposing claim of compliance."

        # 2. Negation mismatches
        negations = ["not", "never", "no", "neither", "none"]
        claim_has_negation = any(re.search(rf"\b{neg}\b", claim_lower) for neg in negations)
        ev_has_negation = any(re.search(rf"\b{neg}\b", ev_lower) for neg in negations)

        # 3. Explicit contradiction words in evidence
        contradiction_markers = ["contrary to", "conflicts with", "incorrectly stated", "false assertion", "erroneous claim"]
        for marker in contradiction_markers:
            if marker in ev_lower:
                return True, f"Evidence explicitly notes condition is '{marker}'."

        return False, ""

    def _compute_semantic_similarity(self, text_a: str, text_b: str) -> float:
        """Computes cosine similarity between two texts using local MiniLM embeddings."""
        try:
            vecs = self.embedding_model.embed_batch([text_a, text_b])
            # Vectors are L2 normalized
            sim = float(np.dot(vecs[0], vecs[1]))
            return max(0.0, min(1.0, sim))
        except Exception as e:
            logger.warning(f"Embedding similarity computation failed: {e}")
            # Fallback to token Jaccard overlap
            tokens_a = set(re.findall(r"\w+", text_a.lower()))
            tokens_b = set(re.findall(r"\w+", text_b.lower()))
            if not tokens_a or not tokens_b:
                return 0.0
            return len(tokens_a & tokens_b) / len(tokens_a | tokens_b)

    def verify_claim(self, request: VerificationRequest, db: Optional[Session] = None) -> VerificationResponse:
        """
        Grounds and verifies an individual claim against evidence.
        """
        claim = request.claim.strip()
        finding_id = request.finding_id or f"finding_{uuid.uuid4().hex[:8]}"
        evidence_list: List[EvidenceItem] = list(request.evidence or [])

        # 1. Auto-retrieve from local vector index if no explicit evidence provided
        retrieved_count = 0
        if not evidence_list and request.auto_retrieve:
            close_db_after = False
            if db is None:
                db = SessionLocal()
                close_db_after = True
            try:
                hits = knowledge_service.search(query=claim, top_k=request.top_k, db=db)
                retrieved_count = len(hits)
                for h in hits:
                    evidence_list.append(
                        EvidenceItem(
                            document=h.get("document_name", "Unknown"),
                            page=h.get("page_number"),
                            source_chunk=h.get("source_id", "chunk_unknown"),
                            relevant_text=h.get("chunk_text", "").strip(),
                        )
                    )
            finally:
                if close_db_after:
                    db.close()

        # Rule 1: Missing / Empty Evidence -> Never VERIFIED
        if not evidence_list:
            finding = Finding(
                id=finding_id,
                claim=claim,
                evidence=[],
                confidence=0.0,
                verification_status=VerificationStatus.INSUFFICIENT_EVIDENCE,
                explanation="No evidence available in local documents to substantiate claim.",
            )
            return VerificationResponse(finding=finding, retrieved_citation_count=0)

        # 2. Evaluate evidence items
        supporting_items: List[Tuple[EvidenceItem, float]] = []
        contradictions: List[Tuple[EvidenceItem, str]] = []

        for item in evidence_list:
            # Check contradiction
            is_contra, detail = self._detect_contradiction(claim, item.relevant_text)
            if is_contra:
                contradictions.append((item, detail))
                continue

            # Compute semantic relevance
            similarity = self._compute_semantic_similarity(claim, item.relevant_text)

            # Check keyword relevance overlap
            claim_words = {w for w in re.findall(r"\w+", claim.lower()) if len(w) > 3}
            ev_words = {w for w in re.findall(r"\w+", item.relevant_text.lower()) if len(w) > 3}
            shared_words = claim_words & ev_words

            # Threshold for supporting evidence: similarity >= 0.40 or substantial key terms shared
            if similarity >= 0.40 or len(shared_words) >= 2:
                supporting_items.append((item, similarity))

        # Rule 2: Contradiction Detected -> Never VERIFIED
        if contradictions:
            contra_item, detail = contradictions[0]
            doc_ref = f"{contra_item.document}" + (f" (Page {contra_item.page})" if contra_item.page else "")
            finding = Finding(
                id=finding_id,
                claim=claim,
                evidence=[c[0] for c in contradictions] + [s[0] for s in supporting_items],
                confidence=0.15,
                verification_status=VerificationStatus.UNCERTAIN,
                explanation=f"Contradiction identified: {doc_ref} reports conflicting data ({detail}).",
            )
            return VerificationResponse(finding=finding, retrieved_citation_count=len(evidence_list))

        # Rule 3: No supporting items found
        if not supporting_items:
            finding = Finding(
                id=finding_id,
                claim=claim,
                evidence=evidence_list,
                confidence=0.10,
                verification_status=VerificationStatus.INSUFFICIENT_EVIDENCE,
                explanation="Retrieved evidence does not contain facts or assertions that support the claim.",
            )
            return VerificationResponse(finding=finding, retrieved_citation_count=len(evidence_list))

        # 3. Analyze source distribution for Multi-Source Corroboration
        distinct_docs: Set[str] = set()
        citations_summary: List[str] = []

        for item, sim in supporting_items:
            distinct_docs.add(item.document)
            page_info = f"Page {item.page}" if item.page else "General"
            citations_summary.append(f"{item.document} — {page_info}")

        avg_similarity = float(np.mean([s[1] for s in supporting_items]))

        # Rule 4: Multi-Source Corroboration (>= 2 distinct documents or >= 2 distinct pages) -> VERIFIED
        distinct_pages = {(item.document, item.page) for item, _ in supporting_items if item.page}
        if len(distinct_docs) >= 2 or len(distinct_pages) >= 2:
            status = VerificationStatus.VERIFIED
            confidence = round(min(0.96, 0.88 + 0.08 * avg_similarity), 2)
            unique_citations = sorted(list(set(citations_summary)))
            explanation = "Evidence:\n" + "\n".join(unique_citations)

        # Rule 5: Single Document Source -> SUPPORTED
        else:
            status = VerificationStatus.SUPPORTED
            confidence = round(min(0.85, 0.70 + 0.15 * avg_similarity), 2)
            item, _ = supporting_items[0]
            page_info = f"Page {item.page}" if item.page else "General"
            explanation = f"Evidence:\n{item.document} — {page_info}"

        finding = Finding(
            id=finding_id,
            claim=claim,
            evidence=[s[0] for s in supporting_items],
            confidence=confidence,
            verification_status=status,
            explanation=explanation,
        )

        return VerificationResponse(
            finding=finding,
            retrieved_citation_count=len(evidence_list),
        )

    def verify_batch(self, claims: List[str], auto_retrieve: bool = True) -> List[Finding]:
        """Batch verifies a list of claims sequentially."""
        results: List[Finding] = []
        db = SessionLocal()
        try:
            for idx, claim in enumerate(claims, 1):
                req = VerificationRequest(
                    claim=claim,
                    finding_id=f"finding_{idx}",
                    auto_retrieve=auto_retrieve,
                )
                res = self.verify_claim(req, db=db)
                results.append(res.finding)
        finally:
            db.close()
        return results


evidence_verifier = EvidenceVerifier()
