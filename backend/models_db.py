import datetime
from sqlalchemy import Column, Integer, String, DateTime, Text, ForeignKey, BigInteger, LargeBinary
from sqlalchemy.orm import relationship
from backend.database import Base


class SystemMetadata(Base):
    """Stores persistent system state and startup history."""
    __tablename__ = "system_metadata"

    id = Column(Integer, primary_key=True, index=True)
    key = Column(String(128), unique=True, nullable=False, index=True)
    value = Column(Text, nullable=False)
    updated_at = Column(
        DateTime,
        default=lambda: datetime.datetime.now(datetime.timezone.utc),
        onupdate=lambda: datetime.datetime.now(datetime.timezone.utc),
    )


class HealthLog(Base):
    """Audit log of health status evaluations."""
    __tablename__ = "health_logs"

    id = Column(Integer, primary_key=True, index=True)
    backend_status = Column(String(32), nullable=False)
    database_status = Column(String(32), nullable=False)
    ai_models_status = Column(String(32), nullable=False)
    timestamp = Column(
        DateTime,
        default=lambda: datetime.datetime.now(datetime.timezone.utc),
        index=True,
    )


class DocumentRecord(Base):
    """Stores metadata for all locally ingested documents."""
    __tablename__ = "documents"

    id = Column(String(64), primary_key=True, index=True)  # UUID string
    filename = Column(String(256), nullable=False, index=True)
    file_type = Column(String(16), nullable=False, index=True)  # pdf, txt, docx, png, jpg, jpeg
    file_size_bytes = Column(BigInteger, nullable=False)
    sha256_hash = Column(String(64), nullable=False, index=True)
    mime_type = Column(String(64), nullable=False)
    local_path = Column(String(512), nullable=False)
    page_count = Column(Integer, default=1, nullable=False)
    ocr_status = Column(String(64), default="NOT_APPLICABLE", nullable=False)
    created_at = Column(
        DateTime,
        default=lambda: datetime.datetime.now(datetime.timezone.utc),
        index=True,
    )

    chunks = relationship(
        "DocumentChunk",
        back_populates="document",
        cascade="all, delete-orphan",
        order_by="DocumentChunk.chunk_index",
    )
    embeddings = relationship(
        "ChunkEmbeddingRecord",
        back_populates="document",
        cascade="all, delete-orphan",
    )


class DocumentChunk(Base):
    """Stores granular, page-traceable text chunks extracted from documents."""
    __tablename__ = "document_chunks"

    id = Column(String(64), primary_key=True, index=True)
    document_id = Column(String(64), ForeignKey("documents.id", ondelete="CASCADE"), nullable=False, index=True)
    chunk_index = Column(Integer, nullable=False)
    filename = Column(String(256), nullable=False)
    page_number = Column(Integer, nullable=True, index=True)  # 1-indexed for PDF; None for non-paged
    source_location = Column(String(128), nullable=False)  # e.g., "page:1, chunk:0" or "paragraph:3"
    text = Column(Text, nullable=False)
    char_count = Column(Integer, nullable=False)
    word_count = Column(Integer, nullable=False)

    document = relationship("DocumentRecord", back_populates="chunks")
    embedding = relationship(
        "ChunkEmbeddingRecord",
        back_populates="chunk",
        uselist=False,
        cascade="all, delete-orphan",
    )


class ChunkEmbeddingRecord(Base):
    """Stores 384-dimensional dense embeddings for vector search."""
    __tablename__ = "chunk_embeddings"

    id = Column(String(64), primary_key=True, index=True)  # matches chunk_id
    chunk_id = Column(String(64), ForeignKey("document_chunks.id", ondelete="CASCADE"), unique=True, nullable=False, index=True)
    document_id = Column(String(64), ForeignKey("documents.id", ondelete="CASCADE"), nullable=False, index=True)
    embedding_dim = Column(Integer, default=384, nullable=False)
    embedding_vector = Column(LargeBinary, nullable=False)  # packed np.float32 binary
    created_at = Column(
        DateTime,
        default=lambda: datetime.datetime.now(datetime.timezone.utc),
        index=True,
    )

    chunk = relationship("DocumentChunk", back_populates="embedding")
    document = relationship("DocumentRecord", back_populates="embeddings")
