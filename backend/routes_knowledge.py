from typing import Any, Dict, List, Optional
from fastapi import APIRouter, Depends, status
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from backend.database import get_db
from backend.knowledge.service import knowledge_service
from backend.logger import logger

router = APIRouter(prefix="/knowledge", tags=["Knowledge Retrieval"])


class SearchRequest(BaseModel):
    """Schema for knowledge search queries."""
    query: str = Field(..., min_length=1, description="Semantic search query text.")
    top_k: int = Field(5, ge=1, le=50, description="Maximum number of relevant chunks to retrieve.")


@router.post("/index/{document_id}", status_code=status.HTTP_200_OK)
def index_document(document_id: str, db: Session = Depends(get_db)):
    """
    Generates local dense embeddings for all chunks in the specified document
    using the verified all-MiniLM-L6-v2 ONNX model and indexes them in SQLite.
    """
    logger.info(f"Received indexing request for document: {document_id}")
    result = knowledge_service.index_document(document_id, db)
    return {
        "status": "success",
        **result,
    }


@router.post("/search", status_code=status.HTTP_200_OK)
def search_knowledge(request: SearchRequest, db: Session = Depends(get_db)):
    """
    Performs offline semantic retrieval across all indexed document chunks.
    Returns ranked citations preserving document name, page number, and source location.
    """
    logger.info(f"Knowledge search: '{request.query}' (top_k={request.top_k})")
    hits = knowledge_service.search(query=request.query, top_k=request.top_k, db=db)
    return {
        "query": request.query,
        "total_results": len(hits),
        "results": hits,
    }
