import os
import shutil
import uuid
from pathlib import Path
from typing import List, Optional
from sqlalchemy.orm import Session

from backend.config import settings
from backend.errors import NexusException
from backend.ingestion.validator import DocumentValidator, DocumentValidationError
from backend.ingestion.extractors import get_extractor
from backend.models_db import DocumentRecord, DocumentChunk
from backend.logger import logger

DOCUMENTS_STORAGE_DIR = settings.DATA_DIR / "documents"


class DocumentNotFoundError(NexusException):
    """Raised when a requested document ID does not exist."""
    def __init__(self, doc_id: str):
        super().__init__(message=f"Document with ID '{doc_id}' not found.", status_code=404)


class DocumentIngestionService:
    """Orchestrates local document storage, extraction, and database persistence."""

    def __init__(self, storage_dir: Optional[Path] = None):
        self.storage_dir = storage_dir or DOCUMENTS_STORAGE_DIR
        self.storage_dir.mkdir(parents=True, exist_ok=True)

    def ingest_document(self, filename: str, content: bytes, db: Session) -> DocumentRecord:
        """
        Validates, locally persists, extracts text/metadata, and stores document records.
        All data remains strictly on the local filesystem.
        """
        # 1. Validation
        val = DocumentValidator.validate(filename=filename, content=content)

        # 2. Allocate Unique Document ID & Directory
        doc_id = str(uuid.uuid4())
        doc_folder = self.storage_dir / doc_id
        doc_folder.mkdir(parents=True, exist_ok=True)

        # 3. Persist file locally
        sanitized_filename = Path(filename).name
        local_file_path = doc_folder / sanitized_filename
        with open(local_file_path, "wb") as f:
            f.write(content)

        logger.info(f"Persisted document locally: {local_file_path} ({val.file_size_bytes} bytes)")

        # 4. Extract content and metadata
        try:
            extractor = get_extractor(val.file_type)
            extraction = extractor.extract(
                document_id=doc_id,
                filename=sanitized_filename,
                content=content,
            )
        except Exception as e:
            # Clean up local file on extraction failure
            shutil.rmtree(doc_folder, ignore_errors=True)
            logger.error(f"Extraction failed for {filename}: {e}", exc_info=True)
            raise DocumentValidationError(f"Failed to process corrupted or malformed document '{filename}': {str(e)}")

        # 5. Persist to SQLite Database
        doc_record = DocumentRecord(
            id=doc_id,
            filename=sanitized_filename,
            file_type=val.file_type,
            file_size_bytes=val.file_size_bytes,
            sha256_hash=val.sha256_hash,
            mime_type=val.mime_type,
            local_path=str(local_file_path),
            page_count=extraction.page_count,
            ocr_status=extraction.ocr_status,
        )
        db.add(doc_record)

        # Add chunks
        for chunk in extraction.chunks:
            db_chunk = DocumentChunk(
                id=chunk.chunk_id,
                document_id=doc_id,
                chunk_index=chunk.chunk_index,
                filename=chunk.filename,
                page_number=chunk.page_number,
                source_location=chunk.source_location,
                text=chunk.text,
                char_count=chunk.char_count,
                word_count=chunk.word_count,
            )
            db.add(db_chunk)

        db.commit()
        db.refresh(doc_record)

        logger.info(
            f"Successfully ingested '{filename}' -> ID: {doc_id} | Pages: {doc_record.page_count} | Chunks: {len(doc_record.chunks)}"
        )
        return doc_record

    def list_documents(self, db: Session) -> List[DocumentRecord]:
        """Returns all ingested documents ordered by creation date descending."""
        return db.query(DocumentRecord).order_by(DocumentRecord.created_at.desc()).all()

    def get_document(self, doc_id: str, db: Session) -> DocumentRecord:
        """Retrieves a specific document and its extracted chunks by ID."""
        doc = db.query(DocumentRecord).filter(DocumentRecord.id == doc_id).first()
        if not doc:
            raise DocumentNotFoundError(doc_id)
        return doc

    def delete_document(self, doc_id: str, db: Session) -> bool:
        """Deletes document from database (with cascade) and purges local disk files."""
        doc = db.query(DocumentRecord).filter(DocumentRecord.id == doc_id).first()
        if not doc:
            raise DocumentNotFoundError(doc_id)

        # 1. Purge local filesystem directory
        doc_folder = self.storage_dir / doc_id
        if doc_folder.exists():
            shutil.rmtree(doc_folder, ignore_errors=True)
            logger.info(f"Purged local storage directory: {doc_folder}")

        # 2. Delete database record
        db.delete(doc)
        db.commit()
        logger.info(f"Deleted document record {doc_id} from SQLite.")
        return True


ingestion_service = DocumentIngestionService()
