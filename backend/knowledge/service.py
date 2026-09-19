import time
from typing import Any, Dict, List, Optional
import numpy as np
from sqlalchemy.orm import Session

from backend.errors import NexusException
from backend.models_db import DocumentRecord, DocumentChunk, ChunkEmbeddingRecord
from backend.models_local.minilm_embedding import local_embedding_model
from backend.logger import logger


class KnowledgeRetrievalService:
    """
    Manages local vector indexing and top-k semantic retrieval using all-MiniLM-L6-v2.
    Stores and computes embeddings 100% locally with zero cloud dependencies.
    """

    def __init__(self, embedding_model=None):
        self.embedding_model = embedding_model or local_embedding_model

    def index_document(self, document_id: str, db: Session) -> Dict[str, Any]:
        """
        Extracts all chunks for a document, generates local embeddings,
        and saves packed float32 vector blobs to SQLite.
        """
        start_time = time.perf_counter()

        # 1. Fetch document and chunks
        doc = db.query(DocumentRecord).filter(DocumentRecord.id == document_id).first()
        if not doc:
            raise NexusException(f"Document with ID '{document_id}' not found.", status_code=404)

        chunks = db.query(DocumentChunk).filter(DocumentChunk.document_id == document_id).order_by(DocumentChunk.chunk_index).all()
        if not chunks:
            return {
                "document_id": document_id,
                "document_name": doc.filename,
                "chunks_indexed": 0,
                "message": "Document contains no text chunks to index.",
                "duration_ms": round((time.perf_counter() - start_time) * 1000, 2),
            }

        texts = [chunk.text for chunk in chunks]
        logger.info(f"Generating local embeddings for {len(texts)} chunks of document '{doc.filename}'...")

        # 2. Generate embeddings via local ONNX model
        vectors = self.embedding_model.embed_batch(texts)  # Shape: [N, 384]

        # 3. Store in SQLite chunk_embeddings table
        # Delete prior embeddings for this document if re-indexing
        db.query(ChunkEmbeddingRecord).filter(ChunkEmbeddingRecord.document_id == document_id).delete()

        for chunk, vector in zip(chunks, vectors):
            vector_blob = vector.astype(np.float32).tobytes()
            embedding_record = ChunkEmbeddingRecord(
                id=chunk.id,
                chunk_id=chunk.id,
                document_id=document_id,
                embedding_dim=384,
                embedding_vector=vector_blob,
            )
            db.add(embedding_record)

        db.commit()

        duration_ms = round((time.perf_counter() - start_time) * 1000, 2)
        logger.info(f"Successfully indexed {len(chunks)} chunks for '{doc.filename}' in {duration_ms}ms")

        return {
            "document_id": document_id,
            "document_name": doc.filename,
            "chunks_indexed": len(chunks),
            "embedding_dim": 384,
            "duration_ms": duration_ms,
        }

    def search(self, query: str, top_k: int, db: Session) -> List[Dict[str, Any]]:
        """
        Embeds query text locally and performs exact dot-product cosine similarity
        against all stored chunk embeddings. Returns top-k citations.
        """
        if not query or not query.strip():
            return []

        clean_top_k = max(1, min(top_k, 50))

        # 1. Fetch all embeddings and join with chunk & document records
        results = (
            db.query(
                ChunkEmbeddingRecord.embedding_vector,
                DocumentChunk.id.label("chunk_id"),
                DocumentChunk.text.label("chunk_text"),
                DocumentChunk.filename.label("document_name"),
                DocumentChunk.page_number.label("page_number"),
                DocumentChunk.source_location.label("source_location"),
                DocumentChunk.document_id.label("document_id"),
            )
            .join(DocumentChunk, ChunkEmbeddingRecord.chunk_id == DocumentChunk.id)
            .all()
        )

        if not results:
            return []

        # 2. Embed query locally
        query_vec = self.embedding_model.embed_batch([query])[0]  # Shape: [384]

        # 3. Stack all chunk vectors into a [N, 384] matrix
        vector_list = [np.frombuffer(row[0], dtype=np.float32) for row in results]
        matrix = np.vstack(vector_list)  # Shape: [N, 384]

        # 4. Compute dot products (cosine similarity since vectors are L2-normalized)
        similarity_scores = np.dot(matrix, query_vec)

        # 5. Retrieve top-k indices
        top_k_count = min(clean_top_k, len(similarity_scores))
        top_indices = np.argsort(similarity_scores)[::-1][:top_k_count]

        # 6. Format search response with full required provenance
        hits: List[Dict[str, Any]] = []
        for idx in top_indices:
            row = results[idx]
            score = float(similarity_scores[idx])
            hits.append({
                "chunk_text": row.chunk_text,
                "document_name": row.document_name,
                "page_number": row.page_number,
                "similarity_score": round(score, 4),
                "source_id": row.chunk_id,
                "source_location": row.source_location,
                "document_id": row.document_id,
            })

        return hits


knowledge_service = KnowledgeRetrievalService()
