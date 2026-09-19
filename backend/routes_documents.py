from typing import Any, Dict, List
from fastapi import APIRouter, Depends, File, UploadFile, status
from sqlalchemy.orm import Session

from backend.database import get_db
from backend.ingestion.service import ingestion_service
from backend.logger import logger

router = APIRouter(prefix="/documents", tags=["Documents"])


def serialize_chunk(chunk) -> Dict[str, Any]:
    return {
        "chunk_id": chunk.id,
        "document_id": chunk.document_id,
        "chunk_index": chunk.chunk_index,
        "filename": chunk.filename,
        "page_number": chunk.page_number,
        "source_location": chunk.source_location,
        "text": chunk.text,
        "char_count": chunk.char_count,
        "word_count": chunk.word_count,
    }


def serialize_document(doc, include_chunks: bool = False) -> Dict[str, Any]:
    data = {
        "id": doc.id,
        "filename": doc.filename,
        "file_type": doc.file_type,
        "file_size_bytes": doc.file_size_bytes,
        "sha256_hash": doc.sha256_hash,
        "mime_type": doc.mime_type,
        "page_count": doc.page_count,
        "ocr_status": doc.ocr_status,
        "chunk_count": len(doc.chunks) if doc.chunks else 0,
        "created_at": doc.created_at.isoformat() if doc.created_at else None,
    }
    if include_chunks and doc.chunks:
        data["chunks"] = [serialize_chunk(c) for c in doc.chunks]
    return data


@router.post("", status_code=status.HTTP_201_CREATED)
async def upload_document(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    """
    Ingests and processes a local document (PDF, TXT, DOCX, PNG, JPG, JPEG).
    Extracts page-level chunks, stores files strictly locally, and updates SQLite.
    """
    logger.info(f"Received document upload: {file.filename} ({file.content_type})")
    content = await file.read()
    doc_record = ingestion_service.ingest_document(
        filename=file.filename or "unnamed_document",
        content=content,
        db=db,
    )
    return {
        "status": "success",
        "message": f"Document '{doc_record.filename}' successfully ingested locally.",
        "document": serialize_document(doc_record, include_chunks=True),
    }


@router.get("", status_code=status.HTTP_200_OK)
def list_documents(db: Session = Depends(get_db)):
    """Lists all locally ingested documents."""
    docs = ingestion_service.list_documents(db)
    return {
        "total_documents": len(docs),
        "documents": [serialize_document(d, include_chunks=False) for d in docs],
    }


@router.get("/{document_id}", status_code=status.HTTP_200_OK)
def get_document(document_id: str, db: Session = Depends(get_db)):
    """Retrieves document metadata along with all granular page-level chunks."""
    doc = ingestion_service.get_document(document_id, db)
    return {
        "document": serialize_document(doc, include_chunks=True),
    }


@router.delete("/{document_id}", status_code=status.HTTP_200_OK)
def delete_document(document_id: str, db: Session = Depends(get_db)):
    """Deletes document record from SQLite and purges local disk files."""
    ingestion_service.delete_document(document_id, db)
    return {
        "status": "success",
        "message": f"Document '{document_id}' and all associated local files have been deleted.",
        "deleted_id": document_id,
    }
