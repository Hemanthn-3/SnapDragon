"""
NEXUS Phase 3: RAG Retrieval Quality Evaluation Benchmark
Measures Recall@K, Precision@K, MRR, and Search Latency across a labeled test set.
"""

import os
import sys
import time
import json
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlalchemy.orm import Session
from backend.database import SessionLocal, init_database
from backend.models_db import DocumentRecord
from backend.ingestion.service import ingestion_service
from backend.knowledge.service import knowledge_service
from tests.fixtures_data import create_synthetic_pdf, create_synthetic_docx, create_synthetic_txt


def run_evaluation():
    init_database()
    db: Session = SessionLocal()

    # Reset and seed benchmark documents
    db.query(DocumentRecord).delete()
    db.commit()

    doc_pdf = ingestion_service.ingest_document("Snapdragon_SoC_Architecture.pdf", create_synthetic_pdf(3), db)
    doc_docx = ingestion_service.ingest_document("NEXUS_Security_Whitepaper.docx", create_synthetic_docx(), db)
    doc_txt = ingestion_service.ingest_document("NEXUS_Ingestion_Policy.txt", create_synthetic_txt(), db)

    # Index all documents
    knowledge_service.index_document(doc_pdf.id, db)
    knowledge_service.index_document(doc_docx.id, db)
    knowledge_service.index_document(doc_txt.id, db)

    test_cases = [
        {
            "id": "Q01",
            "query": "What is the dedicated tensor acceleration capability of the Qualcomm Hexagon NPU?",
            "target_doc": "Snapdragon_SoC_Architecture.pdf",
            "target_page": 2,
            "target_keyword": "45 TOPS",
        },
        {
            "id": "Q02",
            "query": "How many cores does the Qualcomm Oryon CPU feature and what is its clock frequency?",
            "target_doc": "Snapdragon_SoC_Architecture.pdf",
            "target_page": 1,
            "target_keyword": "12-core",
        },
        {
            "id": "Q03",
            "query": "What is the compute capability of the Adreno graphics processor?",
            "target_doc": "Snapdragon_SoC_Architecture.pdf",
            "target_page": 3,
            "target_keyword": "4.6 TFLOPS",
        },
        {
            "id": "Q04",
            "query": "Are network calls and telemetry permitted in the local ingestion subsystem?",
            "target_doc": "NEXUS_Ingestion_Policy.txt",
            "target_page": None,
            "target_keyword": "telemetry",
        },
        {
            "id": "Q05",
            "query": "Which speech model is designated in the architecture table for audio transcription?",
            "target_doc": "NEXUS_Security_Whitepaper.docx",
            "target_page": None,
            "target_keyword": "Whisper-Small",
        },
        {
            "id": "Q06",
            "query": "Which vision encoder model is specified in the architecture table?",
            "target_doc": "NEXUS_Security_Whitepaper.docx",
            "target_page": None,
            "target_keyword": "CLIP ViT-B/32",
        },
        {
            "id": "Q07",
            "query": "What metadata is retained on extracted chunks for deterministic source attribution?",
            "target_doc": "NEXUS_Ingestion_Policy.txt",
            "target_page": None,
            "target_keyword": "deterministic",
        },
        {
            "id": "Q08",
            "query": "What quantization is used for neural networks on the Hexagon tensor engine?",
            "target_doc": "Snapdragon_SoC_Architecture.pdf",
            "target_page": 2,
            "target_keyword": "quantized",
        },
        {
            "id": "Q09",
            "query": "How are multi-monitor displays driven on the Snapdragon platform?",
            "target_doc": "Snapdragon_SoC_Architecture.pdf",
            "target_page": 3,
            "target_keyword": "display processor",
        },
        {
            "id": "Q10",
            "query": "Does NEXUS operate as an offline-first work agent?",
            "target_doc": "NEXUS_Security_Whitepaper.docx",
            "target_page": None,
            "target_keyword": "offline-first",
        },
    ]

    eval_results = []
    top_1_hits = 0
    top_3_hits = 0
    reciprocal_ranks = []
    latencies = []

    for tc in test_cases:
        t0 = time.perf_counter()
        hits = knowledge_service.search(tc["query"], top_k=3, db=db)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        latencies.append(elapsed_ms)

        rank = 0
        hit_doc = None
        hit_page = None
        score = 0.0

        for idx, h in enumerate(hits):
            doc_match = h["document_name"] == tc["target_doc"]
            page_match = (tc["target_page"] is None) or (h["page_number"] == tc["target_page"])
            keyword_match = tc["target_keyword"].lower() in h["chunk_text"].lower()

            if doc_match and page_match and keyword_match:
                rank = idx + 1
                hit_doc = h["document_name"]
                hit_page = h["page_number"]
                score = h["similarity_score"]
                break

        if rank == 1:
            top_1_hits += 1
        if 0 < rank <= 3:
            top_3_hits += 1

        rr = 1.0 / rank if rank > 0 else 0.0
        reciprocal_ranks.append(rr)

        target_str = tc["target_doc"] + (f" (page {tc['target_page']})" if tc["target_page"] else "")
        retrieved_str = (hit_doc + (f" (page {hit_page})" if hit_page else "")) if rank > 0 else "MISS"

        eval_results.append({
            "id": tc["id"],
            "query": tc["query"],
            "target": target_str,
            "retrieved": retrieved_str,
            "rank": rank if rank > 0 else "MISS",
            "similarity_score": score,
            "latency_ms": round(elapsed_ms, 2),
        })

    p_at_1 = top_1_hits / len(test_cases)
    r_at_3 = top_3_hits / len(test_cases)
    mrr = sum(reciprocal_ranks) / len(test_cases)
    avg_lat = sum(latencies) / len(latencies)

    report = {
        "total_queries": len(test_cases),
        "recall_at_1": round(p_at_1 * 100, 1),
        "recall_at_3": round(r_at_3 * 100, 1),
        "precision_at_1": round(p_at_1 * 100, 1),
        "mrr": round(mrr, 4),
        "avg_latency_ms": round(avg_lat, 2),
        "cases": eval_results,
    }

    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    run_evaluation()
