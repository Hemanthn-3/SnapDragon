"""
NEXUS Phase 9: Multimodal Vision API Routes
Provides endpoints for local image inspection and visual understanding.
Distinguishes OBSERVED from INFERRED findings. Zero unrestricted camera monitoring.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from backend.database import get_db
from backend.logger import get_logger
from backend.models_db import DocumentRecord
from backend.models_local.clip_vision import local_clip_vision

logger = get_logger("nexus.routes_vision")

router = APIRouter(prefix="/vision", tags=["Multimodal Vision & Image Understanding"])


class VisionStatusResponse(BaseModel):
    model_name: str
    architecture: Optional[str] = None
    quantization: Optional[str] = None
    target_hardware: str
    current_hardware: Optional[str] = None
    execution_provider: str
    status: str
    is_loaded: bool
    supported_formats: List[str] = Field(default_factory=list)
    blockers: List[str] = Field(default_factory=list)
    epistemic_separation: Dict[str, str] = Field(default_factory=dict)
    capabilities_boundary: Optional[str] = None


class AnalyzeImageRequest(BaseModel):
    document_id: str = Field(..., description="Document ID of an image or PDF")
    page_number: Optional[int] = Field(default=None, ge=1, description="Page number if extracting from a multi-page PDF")
    prompt: Optional[str] = Field(default=None, max_length=500, description="Visual inquiry focus")


class VisionObservationResponse(BaseModel):
    document_id: Optional[str] = None
    filename: str
    status: str
    observed: Dict[str, Any] = Field(..., description="Physical, measurable optical attributes")
    inferred: Dict[str, Any] = Field(..., description="Semantic deductions, category, and scene tags")
    analysis: str
    duration_ms: float
    hardware: str
    execution_provider: str
    extracted_images: Optional[List[Dict[str, Any]]] = None


@router.get(
    "/status",
    response_model=VisionStatusResponse,
    summary="Get Vision Model Status & Diagnostics",
    description="Returns lifecycle status, target hardware, supported formats, and execution provider for OpenAI-CLIP.",
)
def get_vision_status():
    return local_clip_vision.get_status()


@router.post(
    "/analyze",
    response_model=VisionObservationResponse,
    summary="Analyze Uploaded Image or PDF File",
    description="Inspects uploaded PNG, JPG, or PDF files and returns structured observations strictly demarcating OBSERVED from INFERRED.",
)
async def analyze_uploaded_file(
    file: UploadFile = File(...),
    prompt: Optional[str] = Form(default=None),
    page_number: Optional[int] = Form(default=None),
):
    if not file or not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No image file provided.",
        )

    # Validate file extension
    ext = Path(file.filename).suffix.lower().lstrip(".")
    if ext not in ["png", "jpg", "jpeg", "pdf"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file format '{ext}'. Supported formats: png, jpg, jpeg, pdf.",
        )

    content_bytes = await file.read()
    if not content_bytes or len(content_bytes) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty.",
        )

    # 1. Native Image (PNG, JPG, JPEG)
    if ext in ["png", "jpg", "jpeg"]:
        res = local_clip_vision.inspect_image(
            image_bytes=content_bytes,
            filename=file.filename,
            prompt=prompt,
        )
        return VisionObservationResponse(
            filename=file.filename,
            status="VERIFIED",
            observed=res["observed"],
            inferred=res["inferred"],
            analysis=res["analysis"],
            duration_ms=res["duration_ms"],
            hardware=res["hardware"],
            execution_provider=res["execution_provider"],
        )

    # 2. PDF Document: Extract embedded raster images
    elif ext == "pdf":
        extracted_imgs = local_clip_vision.extract_images_from_pdf(
            pdf_bytes=content_bytes,
            page_number=page_number,
        )
        if not extracted_imgs:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"No embedded raster images found in PDF '{file.filename}'" + (f" on page {page_number}." if page_number else "."),
            )

        target_img = extracted_imgs[0]
        res = local_clip_vision.inspect_image(
            image_bytes=target_img["image_bytes"],
            filename=f"{file.filename} ({target_img['image_name']})",
            prompt=prompt,
        )
        res["observed"]["pdf_provenance"] = {
            "source_pdf": file.filename,
            "page_number": target_img["page_number"],
            "image_index": target_img["image_index"],
        }
        summaries = [
            {
                "page_number": img["page_number"],
                "image_index": img["image_index"],
                "image_name": img["image_name"],
                "size_bytes": img["size_bytes"],
                "dimensions": img["dimensions"],
                "format": img["format"],
            }
            for img in extracted_imgs
        ]
        return VisionObservationResponse(
            filename=file.filename,
            status="VERIFIED",
            observed=res["observed"],
            inferred=res["inferred"],
            analysis=res["analysis"],
            duration_ms=res["duration_ms"],
            hardware=res["hardware"],
            execution_provider=res["execution_provider"],
            extracted_images=summaries,
        )


@router.post(
    "/analyze-document",
    response_model=VisionObservationResponse,
    summary="Analyze Document Image by Document ID",
    description="Inspects an ingested document image or PDF by ID and returns structured observations.",
)
def analyze_document_image(
    request: AnalyzeImageRequest,
    db: Session = Depends(get_db),
):
    doc = db.query(DocumentRecord).filter(DocumentRecord.id == request.document_id).first()
    if not doc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document with ID '{request.document_id}' not found.",
        )

    file_path = doc.local_path
    if not file_path:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Physical file path missing for document '{doc.filename}'.",
        )

    p = Path(file_path)
    if not p.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"File not found on disk at {file_path}",
        )

    content_bytes = p.read_bytes()

    if doc.file_type in ["png", "jpg", "jpeg"]:
        res = local_clip_vision.inspect_image(
            image_bytes=content_bytes,
            filename=doc.filename,
            prompt=request.prompt,
        )
        return VisionObservationResponse(
            document_id=doc.id,
            filename=doc.filename,
            status="VERIFIED",
            observed=res["observed"],
            inferred=res["inferred"],
            analysis=res["analysis"],
            duration_ms=res["duration_ms"],
            hardware=res["hardware"],
            execution_provider=res["execution_provider"],
        )
    elif doc.file_type == "pdf":
        extracted_imgs = local_clip_vision.extract_images_from_pdf(
            pdf_bytes=content_bytes,
            page_number=request.page_number,
        )
        if not extracted_imgs:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"No embedded raster images found in PDF '{doc.filename}'" + (f" on page {request.page_number}." if request.page_number else "."),
            )

        target_img = extracted_imgs[0]
        res = local_clip_vision.inspect_image(
            image_bytes=target_img["image_bytes"],
            filename=f"{doc.filename} ({target_img['image_name']})",
            prompt=request.prompt,
        )
        res["observed"]["pdf_provenance"] = {
            "source_pdf": doc.filename,
            "page_number": target_img["page_number"],
            "image_index": target_img["image_index"],
        }
        summaries = [
            {
                "page_number": img["page_number"],
                "image_index": img["image_index"],
                "image_name": img["image_name"],
                "size_bytes": img["size_bytes"],
                "dimensions": img["dimensions"],
                "format": img["format"],
            }
            for img in extracted_imgs
        ]
        return VisionObservationResponse(
            document_id=doc.id,
            filename=doc.filename,
            status="VERIFIED",
            observed=res["observed"],
            inferred=res["inferred"],
            analysis=res["analysis"],
            duration_ms=res["duration_ms"],
            hardware=res["hardware"],
            execution_provider=res["execution_provider"],
            extracted_images=summaries,
        )
    else:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Document '{doc.filename}' (type: {doc.file_type}) is not an image or PDF.",
        )
