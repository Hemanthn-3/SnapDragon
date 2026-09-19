import numpy as np
from fastapi import status
from sqlalchemy.orm import Session

from backend.database import SessionLocal
from backend.models_db import ChunkEmbeddingRecord
from backend.models_local.minilm_embedding import local_embedding_model
from tests.fixtures_data import create_synthetic_pdf, create_synthetic_docx, create_synthetic_txt


def test_minilm_embedding_model():
    """Verifies that the local all-MiniLM-L6-v2 ONNX model produces normalized 384-d vectors with semantic fidelity."""
    # 1. Single embedding test
    text = "Qualcomm Snapdragon X Elite NPU matrix computation."
    vec = local_embedding_model.embed_text(text)
    assert isinstance(vec, list)
    assert len(vec) == 384

    # Verify unit norm
    norm = np.linalg.norm(vec)
    assert np.isclose(norm, 1.0, atol=1e-3)

    # 2. Batch embedding and semantic distance test
    s1 = "Snapdragon Hexagon NPU 45 TOPS"
    s2 = "Qualcomm processor silicon hardware"
    s3 = "Homemade chocolate cookies baking recipe"

    vecs = local_embedding_model.embed_batch([s1, s2, s3])
    assert vecs.shape == (3, 384)

    sim_related = float(np.dot(vecs[0], vecs[1]))
    sim_unrelated = float(np.dot(vecs[0], vecs[2]))
    assert sim_related > sim_unrelated


def test_index_document_endpoint(client):
    """Verifies that POST /knowledge/index/{document_id} indexes chunks and persists embeddings locally."""
    # 1. Ingest PDF
    pdf_bytes = create_synthetic_pdf(num_pages=2)
    ingest_resp = client.post("/documents", files={"file": ("snapdragon_spec.pdf", pdf_bytes, "application/pdf")})
    assert ingest_resp.status_code == status.HTTP_201_CREATED
    doc_id = ingest_resp.json()["document"]["id"]

    # 2. Index document
    index_resp = client.post(f"/knowledge/index/{doc_id}")
    assert index_resp.status_code == status.HTTP_200_OK
    data = index_resp.json()
    assert data["status"] == "success"
    assert data["document_id"] == doc_id
    assert data["document_name"] == "snapdragon_spec.pdf"
    assert data["chunks_indexed"] >= 2
    assert data["embedding_dim"] == 384
    assert data["duration_ms"] > 0

    # 3. Verify embeddings stored in SQLite
    db: Session = SessionLocal()
    try:
        embeddings = db.query(ChunkEmbeddingRecord).filter(ChunkEmbeddingRecord.document_id == doc_id).all()
        assert len(embeddings) == data["chunks_indexed"]
        for emb in embeddings:
            arr = np.frombuffer(emb.embedding_vector, dtype=np.float32)
            assert arr.shape == (384,)
    finally:
        db.close()

    # Cleanup
    client.delete(f"/documents/{doc_id}")


def test_knowledge_search_exact_match(client):
    """Verifies that POST /knowledge/search retrieves the correct source chunk, page number, and citations."""
    # 1. Ingest multi-page PDF
    pdf_bytes = create_synthetic_pdf(num_pages=3)
    doc_resp = client.post("/documents", files={"file": ("benchmark_report.pdf", pdf_bytes, "application/pdf")})
    doc_id = doc_resp.json()["document"]["id"]

    # 2. Index
    client.post(f"/knowledge/index/{doc_id}")

    # 3. Search for specific page 2 content (Hexagon NPU)
    search_payload = {
        "query": "How many TOPS of AI computation does the dedicated Hexagon NPU deliver?",
        "top_k": 3,
    }
    search_resp = client.post("/knowledge/search", json=search_payload)
    assert search_resp.status_code == status.HTTP_200_OK
    search_data = search_resp.json()

    assert search_data["total_results"] > 0
    top_hit = search_data["results"][0]

    # Verify required citation fields
    assert "chunk_text" in top_hit
    assert "document_name" in top_hit
    assert top_hit["document_name"] == "benchmark_report.pdf"
    assert "page_number" in top_hit
    assert top_hit["page_number"] == 2
    assert "similarity_score" in top_hit
    assert top_hit["similarity_score"] > 0.5
    assert "source_id" in top_hit
    assert "source_location" in top_hit
    assert "page:2" in top_hit["source_location"]

    # Cleanup
    client.delete(f"/documents/{doc_id}")


def test_knowledge_search_top_k(client):
    """Verifies that top_k retrieval respects the requested result count across multiple documents."""
    # Ingest 2 distinct documents
    doc1_resp = client.post("/documents", files={"file": ("doc_a.txt", create_synthetic_txt(), "text/plain")})
    doc2_resp = client.post("/documents", files={"file": ("doc_b.docx", create_synthetic_docx(), "application/vnd.openxmlformats-officedocument.wordprocessingml.document")})
    id1 = doc1_resp.json()["document"]["id"]
    id2 = doc2_resp.json()["document"]["id"]

    client.post(f"/knowledge/index/{id1}")
    client.post(f"/knowledge/index/{id2}")

    # Top-K = 2
    resp_2 = client.post("/knowledge/search", json={"query": "multimodal AI work agent", "top_k": 2})
    assert resp_2.status_code == status.HTTP_200_OK
    assert len(resp_2.json()["results"]) == 2

    # Top-K = 5
    resp_5 = client.post("/knowledge/search", json={"query": "multimodal AI work agent", "top_k": 5})
    assert resp_5.status_code == status.HTTP_200_OK
    assert len(resp_5.json()["results"]) <= 5

    # Scores should be sorted descending
    scores = [r["similarity_score"] for r in resp_5.json()["results"]]
    assert scores == sorted(scores, reverse=True)

    # Cleanup
    client.delete(f"/documents/{id1}")
    client.delete(f"/documents/{id2}")


def test_cascade_deletion_purges_embeddings(client):
    """Verifies that deleting a document automatically cascades and removes its vector embeddings."""
    doc_resp = client.post("/documents", files={"file": ("cascade_test.txt", create_synthetic_txt(), "text/plain")})
    doc_id = doc_resp.json()["document"]["id"]

    client.post(f"/knowledge/index/{doc_id}")

    db: Session = SessionLocal()
    try:
        count_before = db.query(ChunkEmbeddingRecord).filter(ChunkEmbeddingRecord.document_id == doc_id).count()
        assert count_before > 0
    finally:
        db.close()

    # Delete document
    client.delete(f"/documents/{doc_id}")

    db2: Session = SessionLocal()
    try:
        count_after = db2.query(ChunkEmbeddingRecord).filter(ChunkEmbeddingRecord.document_id == doc_id).count()
        assert count_after == 0
    finally:
        db2.close()
