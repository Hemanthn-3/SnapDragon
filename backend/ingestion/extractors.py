import io
import uuid
from dataclasses import dataclass, field
from typing import List, Optional
from PIL import Image
import pypdf
import docx

from backend.logger import logger


@dataclass
class ExtractedChunk:
    """Represents an atomic extracted text chunk with full spatial/page provenance."""
    chunk_id: str
    document_id: str
    filename: str
    chunk_index: int
    page_number: Optional[int]  # 1-indexed for PDF; None for unpaged docs
    source_location: str  # e.g., "page:1, chunk:0" or "paragraph:2"
    text: str
    char_count: int
    word_count: int


@dataclass
class ExtractionResult:
    """Overall result of extracting content from a document."""
    page_count: int
    chunks: List[ExtractedChunk] = field(default_factory=list)
    ocr_status: str = "NOT_APPLICABLE"
    notes: Optional[str] = None


class BaseExtractor:
    """Base extractor interface."""
    def extract(self, document_id: str, filename: str, content: bytes) -> ExtractionResult:
        raise NotImplementedError


class PDFExtractor(BaseExtractor):
    """Extracts text from PDF documents page-by-page preserving 1-indexed page provenance."""

    def extract(self, document_id: str, filename: str, content: bytes) -> ExtractionResult:
        stream = io.BytesIO(content)
        reader = pypdf.PdfReader(stream)
        total_pages = len(reader.pages)
        chunks: List[ExtractedChunk] = []
        has_text = False

        chunk_counter = 0
        for page_idx, page in enumerate(reader.pages):
            page_num = page_idx + 1  # 1-indexed
            try:
                page_text = page.extract_text() or ""
            except Exception as e:
                logger.warning(f"Error extracting text from {filename} page {page_num}: {e}")
                page_text = ""

            page_text = page_text.strip()
            if page_text:
                has_text = True
                # Clean and split into coherent paragraph chunks if page is long
                paragraphs = [p.strip() for p in page_text.split("\n\n") if p.strip()]
                if not paragraphs:
                    paragraphs = [page_text]

                for p_idx, para in enumerate(paragraphs):
                    chunk = ExtractedChunk(
                        chunk_id=str(uuid.uuid4()),
                        document_id=document_id,
                        filename=filename,
                        chunk_index=chunk_counter,
                        page_number=page_num,
                        source_location=f"page:{page_num}, chunk:{p_idx}",
                        text=para,
                        char_count=len(para),
                        word_count=len(para.split()),
                    )
                    chunks.append(chunk)
                    chunk_counter += 1

        # Check if this is a scanned PDF with no native digital text
        if not has_text and total_pages > 0:
            logger.info(f"PDF '{filename}' has {total_pages} page(s) but no native digital text (scanned PDF).")
            return ExtractionResult(
                page_count=total_pages,
                chunks=[],
                ocr_status="NOT_YET_IMPLEMENTED",
                notes="Scanned PDF detected. Native text absent. OCR STATUS: NOT YET IMPLEMENTED",
            )

        return ExtractionResult(
            page_count=total_pages,
            chunks=chunks,
            ocr_status="NOT_APPLICABLE" if has_text else "NOT_YET_IMPLEMENTED",
            notes=f"Extracted {len(chunks)} chunk(s) across {total_pages} page(s).",
        )


class DOCXExtractor(BaseExtractor):
    """Extracts text from DOCX documents preserving paragraph and table structure."""

    def extract(self, document_id: str, filename: str, content: bytes) -> ExtractionResult:
        stream = io.BytesIO(content)
        doc = docx.Document(stream)
        chunks: List[ExtractedChunk] = []
        chunk_counter = 0

        # 1. Extract Paragraphs
        for p_idx, paragraph in enumerate(doc.paragraphs):
            text = paragraph.text.strip()
            if text:
                chunk = ExtractedChunk(
                    chunk_id=str(uuid.uuid4()),
                    document_id=document_id,
                    filename=filename,
                    chunk_index=chunk_counter,
                    page_number=None,  # DOCX does not have fixed physical page bounds
                    source_location=f"paragraph:{p_idx + 1}",
                    text=text,
                    char_count=len(text),
                    word_count=len(text.split()),
                )
                chunks.append(chunk)
                chunk_counter += 1

        # 2. Extract Tables
        for t_idx, table in enumerate(doc.tables):
            table_rows = []
            for row in table.rows:
                row_cells = [cell.text.strip() for cell in row.cells if cell.text.strip()]
                if row_cells:
                    table_rows.append(" | ".join(row_cells))

            if table_rows:
                table_text = "\n".join(table_rows)
                chunk = ExtractedChunk(
                    chunk_id=str(uuid.uuid4()),
                    document_id=document_id,
                    filename=filename,
                    chunk_index=chunk_counter,
                    page_number=None,
                    source_location=f"table:{t_idx + 1}",
                    text=table_text,
                    char_count=len(table_text),
                    word_count=len(table_text.split()),
                )
                chunks.append(chunk)
                chunk_counter += 1

        return ExtractionResult(
            page_count=1,
            chunks=chunks,
            ocr_status="NOT_APPLICABLE",
            notes=f"Extracted {len(chunks)} structural chunk(s) from DOCX.",
        )


class TXTExtractor(BaseExtractor):
    """Extracts text from plain text files preserving paragraph/block boundaries."""

    def extract(self, document_id: str, filename: str, content: bytes) -> ExtractionResult:
        try:
            full_text = content.decode("utf-8")
        except UnicodeDecodeError:
            full_text = content.decode("latin-1", errors="replace")

        full_text = full_text.strip()
        chunks: List[ExtractedChunk] = []

        if not full_text:
            return ExtractionResult(page_count=1, chunks=[], ocr_status="NOT_APPLICABLE")

        # Split into double-newline paragraphs
        paragraphs = [p.strip() for p in full_text.split("\n\n") if p.strip()]
        if not paragraphs:
            paragraphs = [full_text]

        chunk_counter = 0
        for p_idx, para in enumerate(paragraphs):
            chunk = ExtractedChunk(
                chunk_id=str(uuid.uuid4()),
                document_id=document_id,
                filename=filename,
                chunk_index=chunk_counter,
                page_number=None,
                source_location=f"paragraph:{p_idx + 1}",
                text=para,
                char_count=len(para),
                word_count=len(para.split()),
            )
            chunks.append(chunk)
            chunk_counter += 1

        return ExtractionResult(
            page_count=1,
            chunks=chunks,
            ocr_status="NOT_APPLICABLE",
            notes=f"Extracted {len(chunks)} paragraph chunk(s) from text file.",
        )


class ImageExtractor(BaseExtractor):
    """
    Handles raster images (PNG, JPG, JPEG).
    Extracts image geometry (dimensions, color mode, format).
    In accordance with system specifications, OCR is NOT faked, and no LLM is substituted.
    Reports OCR STATUS: NOT YET IMPLEMENTED.
    """

    def extract(self, document_id: str, filename: str, content: bytes) -> ExtractionResult:
        stream = io.BytesIO(content)
        with Image.open(stream) as img:
            width, height = img.size
            format_name = img.format
            mode = img.mode

        metadata_description = (
            f"Image file '{filename}' [Format: {format_name}, Resolution: {width}x{height}, Mode: {mode}]."
        )

        logger.info(f"{metadata_description} OCR required for text extraction.")

        # In strict adherence to prompt: DO NOT fake OCR results.
        # Report OCR STATUS: NOT YET IMPLEMENTED.
        return ExtractionResult(
            page_count=1,
            chunks=[],
            ocr_status="NOT_YET_IMPLEMENTED",
            notes=f"{metadata_description} OCR STATUS: NOT YET IMPLEMENTED",
        )


def get_extractor(file_type: str) -> BaseExtractor:
    """Factory returning the appropriate extractor for the given file extension/type."""
    clean_type = file_type.lower().lstrip(".")
    if clean_type == "pdf":
        return PDFExtractor()
    elif clean_type == "docx":
        return DOCXExtractor()
    elif clean_type in ("txt", "json"):
        return TXTExtractor()
    elif clean_type in ("png", "jpg", "jpeg"):
        return ImageExtractor()
    else:
        raise ValueError(f"No extractor registered for file type '{clean_type}'")
