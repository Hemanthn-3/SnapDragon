"""
NEXUS Controlled Tool Execution Layer: Implementations
The 7 registered tools for NEXUS:
1. list_documents
2. read_document
3. search_knowledge
4. analyze_image
5. run_ocr
6. create_report
7. export_report

Strictly sandboxed, no shell access, no arbitrary Python evaluation, no file deletion.
"""

import os
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, field_validator
from PIL import Image

from backend.config import settings
from backend.database import SessionLocal
from backend.logger import logger
from backend.models_db import DocumentRecord, DocumentChunk
from backend.knowledge.service import knowledge_service
from backend.tools.base import BaseTool, ToolContext


# Storage directories
DOCUMENTS_DIR = settings.DATA_DIR / "documents"
REPORTS_DIR = settings.DATA_DIR / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def sanitize_path_param(param_value: str, field_name: str) -> str:
    """
    Sanitizes string parameters intended for file or document lookup.
    Rejects path traversal, null bytes, absolute paths, and command injection characters.
    """
    if not param_value:
        raise ValueError(f"{field_name} cannot be empty.")
    
    # Check for null bytes
    if "\x00" in param_value:
        raise ValueError(f"Malicious null byte detected in {field_name}.")
        
    # Check for path traversal sequences
    if ".." in param_value or "/" in param_value or "\\" in param_value:
        raise ValueError(f"Path traversal sequence detected in {field_name}.")
        
    # Check for command injection characters
    injection_chars = [";", "|", "&", "$", "`", "<", ">", "\n", "\r"]
    for char in injection_chars:
        if char in param_value:
            raise ValueError(f"Invalid character '{char}' in {field_name}.")
            
    return param_value.strip()


# =====================================================================
# 1. Tool: list_documents
# =====================================================================

class DocumentSummary(BaseModel):
    id: str
    filename: str
    file_type: str
    page_count: int
    chunk_count: int
    created_at: Optional[str] = None


class ListDocumentsInput(BaseModel):
    filter_type: Optional[str] = Field(default=None, description="Optional file extension filter, e.g. 'pdf', 'txt'")
    limit: int = Field(default=50, ge=1, le=100, description="Max documents to return")

    @field_validator("filter_type")
    @classmethod
    def clean_filter(cls, v):
        if v:
            return sanitize_path_param(v.lower().strip("."), "filter_type")
        return v


class ListDocumentsOutput(BaseModel):
    documents: List[DocumentSummary]
    total_count: int


class ListDocumentsTool(BaseTool):
    name = "list_documents"
    description = "Lists locally ingested documents and their metadata."
    input_schema = ListDocumentsInput
    output_schema = ListDocumentsOutput
    requires_approval = False
    timeout_seconds = 10.0

    def _run(self, params: ListDocumentsInput, context: ToolContext) -> ListDocumentsOutput:
        db = SessionLocal()
        try:
            query = db.query(DocumentRecord)
            if params.filter_type:
                query = query.filter(DocumentRecord.file_type == params.filter_type)
            docs = query.order_by(DocumentRecord.created_at.desc()).limit(params.limit).all()

            summaries = [
                DocumentSummary(
                    id=d.id,
                    filename=d.filename,
                    file_type=d.file_type,
                    page_count=d.page_count,
                    chunk_count=len(d.chunks) if d.chunks else 0,
                    created_at=d.created_at.isoformat() if d.created_at else None,
                )
                for d in docs
            ]
            return ListDocumentsOutput(documents=summaries, total_count=len(summaries))
        finally:
            db.close()


# =====================================================================
# 2. Tool: read_document
# =====================================================================

class ChunkData(BaseModel):
    chunk_id: str
    page_number: Optional[int] = None
    source_location: str
    text: str


class ReadDocumentInput(BaseModel):
    document_id: str = Field(..., description="Unique document ID (UUID)")
    page_number: Optional[int] = Field(default=None, ge=1, description="Specific page number to extract")

    @field_validator("document_id")
    @classmethod
    def validate_doc_id(cls, v):
        return sanitize_path_param(v, "document_id")


class ReadDocumentOutput(BaseModel):
    document_id: str
    filename: str
    chunks: List[ChunkData]
    chunk_count: int
    content: str


class ReadDocumentTool(BaseTool):
    name = "read_document"
    description = "Reads extracted chunks and content from a locally stored document."
    input_schema = ReadDocumentInput
    output_schema = ReadDocumentOutput
    requires_approval = False
    timeout_seconds = 15.0

    def _run(self, params: ReadDocumentInput, context: ToolContext) -> ReadDocumentOutput:
        db = SessionLocal()
        try:
            doc = db.query(DocumentRecord).filter(DocumentRecord.id == params.document_id).first()
            if not doc:
                raise FileNotFoundError(f"Document with ID '{params.document_id}' not found.")

            # Path sandboxing verification
            doc_path = Path(doc.local_path).resolve()
            sandboxed_dir = DOCUMENTS_DIR.resolve()
            if not str(doc_path).startswith(str(sandboxed_dir)):
                raise PermissionError(f"Access denied: Document path outside sandbox ({doc_path})")

            # Retrieve chunks
            chunk_query = db.query(DocumentChunk).filter(DocumentChunk.document_id == params.document_id)
            if params.page_number is not None:
                chunk_query = chunk_query.filter(DocumentChunk.page_number == params.page_number)
            chunks = chunk_query.order_by(DocumentChunk.chunk_index.asc()).all()

            chunk_data_list = [
                ChunkData(
                    chunk_id=c.id,
                    page_number=c.page_number,
                    source_location=c.source_location,
                    text=c.text,
                )
                for c in chunks
            ]
            full_text = "\n\n".join(c.text for c in chunk_data_list)

            return ReadDocumentOutput(
                document_id=doc.id,
                filename=doc.filename,
                chunks=chunk_data_list,
                chunk_count=len(chunk_data_list),
                content=full_text,
            )
        finally:
            db.close()


# =====================================================================
# 3. Tool: search_knowledge
# =====================================================================

class CitationData(BaseModel):
    chunk_id: str
    document_id: str
    document_name: str
    page_number: Optional[int]
    source_location: str
    similarity_score: float
    text: str


class SearchKnowledgeInput(BaseModel):
    query: str = Field(..., min_length=1, max_length=500, description="Semantic retrieval query")
    top_k: int = Field(default=5, ge=1, le=20, description="Max ranked citations to retrieve")

    @field_validator("query")
    @classmethod
    def clean_query(cls, v):
        if "\x00" in v:
            raise ValueError("Null byte detected in query.")
        return v.strip()


class SearchKnowledgeOutput(BaseModel):
    query: str
    citations: List[CitationData]
    total_citations: int


class SearchKnowledgeTool(BaseTool):
    name = "search_knowledge"
    description = "Searches the local dense vector index for semantically relevant chunks."
    input_schema = SearchKnowledgeInput
    output_schema = SearchKnowledgeOutput
    requires_approval = False
    timeout_seconds = 20.0

    def _run(self, params: SearchKnowledgeInput, context: ToolContext) -> SearchKnowledgeOutput:
        db = SessionLocal()
        try:
            results = knowledge_service.search(query=params.query, top_k=params.top_k, db=db)
            citations = [
                CitationData(
                    chunk_id=r.get("chunk_id", ""),
                    document_id=r.get("document_id", ""),
                    document_name=r.get("document_name", ""),
                    page_number=r.get("page_number"),
                    source_location=r.get("source_location", ""),
                    similarity_score=r.get("similarity_score", 0.0),
                    text=r.get("text", ""),
                )
                for r in results
            ]
            return SearchKnowledgeOutput(
                query=params.query,
                citations=citations,
                total_citations=len(citations),
            )
        finally:
            db.close()


from backend.models_local.clip_vision import local_clip_vision

# =====================================================================
# 4. Tool: analyze_image
# =====================================================================

class AnalyzeImageInput(BaseModel):
    document_id: str = Field(..., description="Document ID of an image or PDF")
    page_number: Optional[int] = Field(default=None, ge=1, description="Page number for PDF embedded image inspection")
    prompt: Optional[str] = Field(default=None, max_length=500, description="Visual inquiry prompt")

    @field_validator("document_id")
    @classmethod
    def validate_doc_id(cls, v):
        return sanitize_path_param(v, "document_id")


class AnalyzeImageOutput(BaseModel):
    document_id: str
    status: str
    analysis: str
    image_info: Dict[str, Any] = Field(default_factory=dict)
    observed: Dict[str, Any] = Field(default_factory=dict)
    inferred: Dict[str, Any] = Field(default_factory=dict)
    extracted_images: Optional[List[Dict[str, Any]]] = None


class AnalyzeImageTool(BaseTool):
    name = "analyze_image"
    description = "Analyzes local image documents (PNG, JPG) and embedded PDF images, separating OBSERVED optical metrics from INFERRED semantics."
    input_schema = AnalyzeImageInput
    output_schema = AnalyzeImageOutput
    requires_approval = False
    timeout_seconds = 15.0

    def _run(self, params: AnalyzeImageInput, context: ToolContext) -> AnalyzeImageOutput:
        db = SessionLocal()
        try:
            doc = db.query(DocumentRecord).filter(DocumentRecord.id == params.document_id).first()
            if not doc:
                raise FileNotFoundError(f"Document with ID '{params.document_id}' not found.")

            if doc.file_type not in ["png", "jpg", "jpeg", "pdf"]:
                raise ValueError(f"Document '{doc.filename}' is not an image or PDF (type: {doc.file_type}).")

            file_path = Path(doc.local_path).resolve()
            if not str(file_path).startswith(str(DOCUMENTS_DIR.resolve())):
                raise PermissionError("Path traversal violation: Image outside document sandbox.")

            if not file_path.exists():
                raise FileNotFoundError(f"Physical file missing at {file_path}")

            if doc.file_type == "pdf":
                with open(file_path, "rb") as f:
                    pdf_bytes = f.read()
                pdf_images = local_clip_vision.extract_images_from_pdf(pdf_bytes, page_number=params.page_number)
                if not pdf_images:
                    return AnalyzeImageOutput(
                        document_id=doc.id,
                        status="NO_IMAGES_FOUND",
                        analysis=f"No embedded raster images found in PDF '{doc.filename}'.",
                        image_info={"filename": doc.filename, "file_type": "pdf", "page_number": params.page_number},
                        observed={"page_number": params.page_number, "embedded_image_count": 0},
                        inferred={
                            "capabilities_boundary": "PDF does not contain extractable raster graphics or is pure text/vector.",
                            "visual_category": "document_pure_text_or_vector",
                        },
                        extracted_images=[],
                    )

                # Analyze the targeted extracted image (first matching or first overall)
                target_img = pdf_images[0]
                res = local_clip_vision.inspect_image(
                    image_input=target_img["image_bytes"],
                    filename=target_img["image_name"],
                    prompt=params.prompt,
                )

                extracted_summaries = [
                    {
                        "page_number": img["page_number"],
                        "image_index": img["image_index"],
                        "image_name": img["image_name"],
                        "size_bytes": img["size_bytes"],
                        "dimensions": img["dimensions"],
                        "format": img["format"],
                    }
                    for img in pdf_images
                ]

                info = {
                    "filename": target_img["image_name"],
                    "width": res["observed"]["dimensions"]["width"],
                    "height": res["observed"]["dimensions"]["height"],
                    "format": res["observed"]["format"],
                    "mode": res["observed"]["color_mode"],
                    "source_pdf": doc.filename,
                    "page_number": target_img["page_number"],
                    "total_pdf_images": len(pdf_images),
                }

                analysis_text = (
                    f"PDF Embedded Image Analysis: Extracted {len(pdf_images)} image(s) from '{doc.filename}'. "
                    f"Inspected {target_img['image_name']} (Page {target_img['page_number']}): {res['inferred']['description']}"
                )

                return AnalyzeImageOutput(
                    document_id=doc.id,
                    status="VERIFIED",
                    analysis=analysis_text,
                    image_info=info,
                    observed=res["observed"],
                    inferred=res["inferred"],
                    extracted_images=extracted_summaries,
                )
            else:
                # Direct PNG / JPG / JPEG image
                res = local_clip_vision.inspect_image(
                    image_input=file_path,
                    filename=doc.filename,
                    prompt=params.prompt,
                )

                info = {
                    "filename": doc.filename,
                    "width": res["observed"]["dimensions"]["width"],
                    "height": res["observed"]["dimensions"]["height"],
                    "format": res["observed"]["format"],
                    "mode": res["observed"]["color_mode"],
                }

                return AnalyzeImageOutput(
                    document_id=doc.id,
                    status="VERIFIED",
                    analysis=f"Image Analysis: {doc.filename}. {res['inferred']['description']}",
                    image_info=info,
                    observed=res["observed"],
                    inferred=res["inferred"],
                )
        finally:
            db.close()


# =====================================================================
# 5. Tool: run_ocr
# =====================================================================

class RunOCRInput(BaseModel):
    document_id: str = Field(..., description="Document ID to process")
    page_number: Optional[int] = Field(default=None, ge=1, description="Page number for OCR")

    @field_validator("document_id")
    @classmethod
    def validate_doc_id(cls, v):
        return sanitize_path_param(v, "document_id")


class RunOCROutput(BaseModel):
    document_id: str
    ocr_status: str
    extracted_text: str


class RunOCRTool(BaseTool):
    name = "run_ocr"
    description = "Executes OCR on scanned documents or images without LLM hallucination."
    input_schema = RunOCRInput
    output_schema = RunOCROutput
    requires_approval = False
    timeout_seconds = 15.0

    def _run(self, params: RunOCRInput, context: ToolContext) -> RunOCROutput:
        db = SessionLocal()
        try:
            doc = db.query(DocumentRecord).filter(DocumentRecord.id == params.document_id).first()
            if not doc:
                raise FileNotFoundError(f"Document with ID '{params.document_id}' not found.")

            file_path = Path(doc.local_path).resolve()
            if not str(file_path).startswith(str(DOCUMENTS_DIR.resolve())):
                raise PermissionError("Path traversal violation: Document outside sandbox.")

            # Per project specification: Do not fake OCR results or substitute LLM for OCR.
            # Report truthful status:
            return RunOCROutput(
                document_id=doc.id,
                ocr_status="OCR STATUS: NOT YET IMPLEMENTED",
                extracted_text="OCR STATUS: NOT YET IMPLEMENTED. Scanned image text extraction requires offline EasyOCR/Tesseract engine.",
            )
        finally:
            db.close()


# =====================================================================
# 6. Tool: create_report
# =====================================================================

class ReportSection(BaseModel):
    title: str = Field(..., min_length=1, max_length=200)
    content: str = Field(..., min_length=1)


class CreateReportInput(BaseModel):
    title: str = Field(..., min_length=1, max_length=200, description="Report headline")
    summary: str = Field(..., min_length=1, description="Executive summary")
    sections: List[ReportSection] = Field(default_factory=list, description="Structured report sections")


class CreateReportOutput(BaseModel):
    report_id: str
    title: str
    content: str
    char_count: int


class CreateReportTool(BaseTool):
    name = "create_report"
    description = "Synthesizes findings into a structured markdown report in-memory."
    input_schema = CreateReportInput
    output_schema = CreateReportOutput
    requires_approval = False
    timeout_seconds = 10.0

    def _run(self, params: CreateReportInput, context: ToolContext) -> CreateReportOutput:
        report_id = f"rep_{uuid.uuid4().hex[:10]}"
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

        lines = [
            f"# {params.title}",
            f"\n> **Generated Offline by NEXUS Work Agent** | *{timestamp}*",
            f"\n## Executive Summary\n\n{params.summary.strip()}",
        ]

        for sec in params.sections:
            lines.append(f"\n## {sec.title.strip()}\n\n{sec.content.strip()}")

        lines.append(
            "\n---\n*Confidential & Local-Only — Generated without cloud connectivity on Snapdragon PC.*"
        )
        full_content = "\n".join(lines)

        return CreateReportOutput(
            report_id=report_id,
            title=params.title,
            content=full_content,
            char_count=len(full_content),
        )


# =====================================================================
# 7. Tool: export_report (REQUIRES APPROVAL & SANDBOXED)
# =====================================================================

class ExportReportInput(BaseModel):
    filename: str = Field(..., min_length=1, max_length=100, description="Destination filename, e.g. 'audit_report.md'")
    content: str = Field(..., min_length=1, description="Report markdown or text content")
    approved: bool = Field(default=False, description="Explicit user approval for file write")

    @field_validator("filename")
    @classmethod
    def validate_filename(cls, v):
        cleaned = v.strip()
        # Disallow null bytes
        if "\x00" in cleaned:
            raise ValueError("Null byte detected in filename.")
        # Disallow path traversal sequences
        if ".." in cleaned or "/" in cleaned or "\\" in cleaned:
            raise ValueError("Path traversal sequence detected in filename.")
        # Disallow command injection characters
        for char in [";", "|", "&", "$", "`", "<", ">", "\n", "\r", ":"]:
            if char in cleaned:
                raise ValueError(f"Invalid character '{char}' in filename.")
        # Disallow dangerous extensions
        ext = Path(cleaned).suffix.lower()
        if ext not in [".md", ".txt", ".json"]:
            raise ValueError(f"Disallowed file extension '{ext}'. Only .md, .txt, and .json permitted.")
        return cleaned


class ExportReportOutput(BaseModel):
    exported_path: str
    filename: str
    bytes_written: int
    status: str


class ExportReportTool(BaseTool):
    name = "export_report"
    description = "Exports report content safely to local data/reports/ directory. Strictly requires user approval."
    input_schema = ExportReportInput
    output_schema = ExportReportOutput
    requires_approval = True  # SECURITY GATE
    timeout_seconds = 10.0

    def _run(self, params: ExportReportInput, context: ToolContext) -> ExportReportOutput:
        # Re-check approval
        if not (context.is_approved or params.approved):
            raise PermissionError("User approval required to export reports to disk.")

        # Ensure sandbox path
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        dest_path = (REPORTS_DIR / params.filename).resolve()

        # Verify destination path is strictly within REPORTS_DIR
        sandboxed_reports = REPORTS_DIR.resolve()
        if not str(dest_path).startswith(str(sandboxed_reports)):
            raise PermissionError("Security violation: Target destination outside reports sandbox.")

        # Safe local write
        content_bytes = params.content.encode("utf-8")
        with open(dest_path, "wb") as f:
            f.write(content_bytes)

        logger.info(f"[EXPORT_REPORT] Successfully exported {len(content_bytes)} bytes to {dest_path}")

        return ExportReportOutput(
            exported_path=str(dest_path),
            filename=params.filename,
            bytes_written=len(content_bytes),
            status="EXPORTED",
        )
