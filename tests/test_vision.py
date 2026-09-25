"""
NEXUS Phase 9 Tests: Local Image Understanding Layer
Validates:
- OpenAI-CLIP visual perception pipeline with strict OBSERVED vs INFERRED epistemic separation
- Supported image formats: JPG, PNG, and raster images extracted from PDFs
- Structured observations (optical dimensions, color stats, luminance, entropy, dominant palette)
- Semantic inference (visual category, tags, description, confidence, capabilities boundary)
- Honest capabilities boundary (distinguishing visual embeddings from fine OCR)
- Integration into the agent tool execution layer (analyze_image tool)
- Zero unrestricted camera monitoring (strict pull model, no webcam daemons/streaming)
- Hardware blocker documentation on non-NPU host environments
- REST API endpoints: GET /vision/status, POST /vision/analyze, POST /vision/analyze-document
"""

import io
import os
import uuid
from pathlib import Path
import pytest
from PIL import Image
import pypdf
import numpy as np

from backend.config import settings
from backend.database import SessionLocal
from backend.models_db import DocumentRecord
from backend.models_local.clip_vision import local_clip_vision
from backend.tools.implementations import (
    AnalyzeImageTool,
    AnalyzeImageInput,
    DOCUMENTS_DIR,
)
from backend.tools.registry import tool_registry


@pytest.fixture
def synthetic_png_bytes():
    """Generates synthetic 120x80 RGB PNG image with known geometric content."""
    img = Image.new("RGB", (120, 80), color=(30, 80, 180))
    # Draw a 40x40 white square in the center
    for x in range(40, 80):
        for y in range(20, 60):
            img.putpixel((x, y), (240, 240, 240))
    bio = io.BytesIO()
    img.save(bio, format="PNG")
    return bio.getvalue()


@pytest.fixture
def synthetic_jpg_bytes():
    """Generates synthetic 100x100 RGB JPEG image."""
    img = Image.new("RGB", (100, 100), color=(200, 100, 50))
    bio = io.BytesIO()
    img.save(bio, format="JPEG")
    return bio.getvalue()


@pytest.fixture
def synthetic_pdf_with_image():
    """Generates a synthetic 1-page PDF containing an embedded 64x64 raster image."""
    img = Image.new("RGB", (64, 64), color=(0, 180, 120))
    bio = io.BytesIO()
    img.save(bio, format="PDF")
    return bio.getvalue()


@pytest.fixture
def synthetic_blank_pdf():
    """Generates a synthetic 1-page PDF without any embedded raster images."""
    writer = pypdf.PdfWriter()
    writer.add_blank_page(width=200, height=200)
    bio = io.BytesIO()
    writer.write(bio)
    return bio.getvalue()


# =====================================================================
# 1. Model Status & Hardware Blocker Tests
# =====================================================================

def test_vision_model_status_and_metadata():
    """Verifies that the vision model reports accurate configuration, formats, and epistemic boundaries."""
    status = local_clip_vision.get_status()
    assert "ResNet" in status["model_name"]
    assert "ResNet-18" in status["architecture"]
    assert status["framework"] == "ONNX Runtime"
    assert status["target_hardware"] == "Snapdragon X Elite Hexagon NPU"
    assert "jpg" in status["supported_formats"]
    assert "png" in status["supported_formats"]
    assert "pdf" in status["supported_formats"]
    assert "observed" in status["epistemic_separation"]
    assert "inferred" in status["epistemic_separation"]


def test_vision_host_blocker_detection_on_development_host():
    """Verifies that running on x86_64 development host documents the exact NPU hardware blocker."""
    status = local_clip_vision.get_status()
    import platform
    if platform.machine().lower() not in ["arm64", "aarch64"]:
        assert len(status["blockers"]) > 0
        assert any("Hexagon HTP" in b or "architecture" in b.lower() for b in status["blockers"])
        # Load and verify CPU execution provider without fake NPU simulation
        local_clip_vision.load()
        loaded_status = local_clip_vision.get_status()
        assert loaded_status["execution_provider"] == "CPUExecutionProvider"


# =====================================================================
# 2. Synthetic PNG & JPG Image Inspection (OBSERVED vs INFERRED)
# =====================================================================

def test_synthetic_png_inspection_strictly_separates_observed_and_inferred(synthetic_png_bytes):
    """
    Verifies that PNG image inspection produces strictly separated OBSERVED and INFERRED dictionaries.
    OBSERVED must contain physical metrics; INFERRED must contain semantic interpretation.
    """
    result = local_clip_vision.inspect_image(
        image_bytes=synthetic_png_bytes,
        filename="audit_diagram.png",
        prompt="Verify layout and color contrast",
    )

    # 1. OBSERVED assertions (objective optical measurements)
    observed = result["observed"]
    assert observed["dimensions"]["width"] == 120
    assert observed["dimensions"]["height"] == 80
    assert observed["aspect_ratio"] == 1.5
    assert observed["format"] == "PNG"
    assert observed["color_mode"] == "RGB"
    assert "channel_statistics" in observed
    assert "red" in observed["channel_statistics"]
    assert "green" in observed["channel_statistics"]
    assert "blue" in observed["channel_statistics"]
    assert 0 <= observed["brightness"] <= 255
    assert observed["contrast"] >= 0
    assert "complexity_entropy" in observed
    assert len(observed["dominant_palette"]) >= 1
    assert "hex" in observed["dominant_palette"][0]
    assert "coverage_percent" in observed["dominant_palette"][0]

    # 2. INFERRED assertions (semantic categorization)
    inferred = result["inferred"]
    assert "visual_category" in inferred
    assert isinstance(inferred["semantic_tags"], list)
    assert 0.0 < inferred["confidence"] <= 1.0
    assert "Verify layout and color contrast" in inferred["description"]

    # 3. Model boundary assertions (must not claim unsupported OCR capabilities)
    boundary = inferred["capabilities_boundary"]
    assert "OCR" in boundary or "alphanumeric" in boundary
    assert "directly from pixel arrays" in boundary


def test_synthetic_jpg_inspection(synthetic_jpg_bytes):
    """Verifies that JPG image inspection works with proper format detection."""
    result = local_clip_vision.inspect_image(
        image_bytes=synthetic_jpg_bytes,
        filename="photo_evidence.jpg",
    )
    assert result["observed"]["format"] == "JPEG"
    assert result["observed"]["dimensions"]["width"] == 100
    assert result["observed"]["dimensions"]["height"] == 100
    assert result["inferred"]["visual_category"] is not None
    assert result["duration_ms"] > 0


# =====================================================================
# 3. PDF Embedded Image Extraction & Inspection
# =====================================================================

def test_extract_images_from_pdf_with_embedded_image(synthetic_pdf_with_image):
    """Verifies extraction of embedded raster images from PDF pages."""
    extracted = local_clip_vision.extract_images_from_pdf(synthetic_pdf_with_image)
    assert len(extracted) == 1
    img = extracted[0]
    assert img["page_number"] == 1
    assert img["image_index"] == 1
    assert img["dimensions"]["width"] == 64
    assert img["dimensions"]["height"] == 64
    assert len(img["image_bytes"]) > 0

    # Inspect the extracted image directly
    res = local_clip_vision.inspect_image(img["image_bytes"], filename=img["image_name"])
    assert res["observed"]["dimensions"]["width"] == 64
    assert res["observed"]["dimensions"]["height"] == 64


def test_extract_images_from_pdf_without_images(synthetic_blank_pdf):
    """Verifies safe handling of PDF containing zero embedded images."""
    extracted = local_clip_vision.extract_images_from_pdf(synthetic_blank_pdf)
    assert extracted == []


# =====================================================================
# 4. Deterministic Visual Embeddings
# =====================================================================

def test_deterministic_visual_embeddings(synthetic_png_bytes):
    """Verifies deterministic 512-dim visual representation generation."""
    emb1 = local_clip_vision.encode_image(synthetic_png_bytes)
    emb2 = local_clip_vision.encode_image(synthetic_png_bytes)
    assert len(emb1) == 512
    assert len(emb2) == 512
    assert emb1 == emb2  # Deterministic representation


# =====================================================================
# 5. Agent Tool Integration: analyze_image
# =====================================================================

def test_analyze_image_tool_on_png_document(synthetic_png_bytes):
    """Verifies analyze_image tool execution on a stored PNG document."""
    db = SessionLocal()
    doc_id = f"doc_img_{uuid.uuid4().hex[:8]}"
    file_path = DOCUMENTS_DIR / f"{doc_id}.png"
    file_path.write_bytes(synthetic_png_bytes)

    try:
        rec = DocumentRecord(
            id=doc_id,
            filename="diagram_spec.png",
            file_type="png",
            file_size_bytes=len(synthetic_png_bytes),
            sha256_hash=uuid.uuid4().hex,
            mime_type="image/png",
            local_path=str(file_path),
        )
        db.add(rec)
        db.commit()

        tool = tool_registry.get_tool("analyze_image")
        result = tool.execute({"document_id": doc_id, "prompt": "Check dimensions"})
        assert result.success is True
        assert result.output["status"] == "VERIFIED"
        assert result.output["image_info"]["width"] == 120
        assert result.output["image_info"]["height"] == 80
        assert "observed" in result.output
        assert result.output["observed"]["aspect_ratio"] == 1.5
        assert "inferred" in result.output
        assert "Check dimensions" in result.output["analysis"]
    finally:
        db.query(DocumentRecord).filter(DocumentRecord.id == doc_id).delete()
        db.commit()
        db.close()
        if file_path.exists():
            file_path.unlink()


def test_analyze_image_tool_on_pdf_with_embedded_image(synthetic_pdf_with_image):
    """Verifies analyze_image tool execution on a PDF containing embedded images."""
    db = SessionLocal()
    doc_id = f"doc_pdf_{uuid.uuid4().hex[:8]}"
    file_path = DOCUMENTS_DIR / f"{doc_id}.pdf"
    file_path.write_bytes(synthetic_pdf_with_image)

    try:
        rec = DocumentRecord(
            id=doc_id,
            filename="technical_drawing.pdf",
            file_type="pdf",
            file_size_bytes=len(synthetic_pdf_with_image),
            sha256_hash=uuid.uuid4().hex,
            mime_type="application/pdf",
            local_path=str(file_path),
        )
        db.add(rec)
        db.commit()

        tool = tool_registry.get_tool("analyze_image")
        result = tool.execute({"document_id": doc_id, "prompt": "Inspect technical drawing"})
        assert result.success is True
        assert result.output["status"] == "VERIFIED"
        assert result.output["image_info"]["total_pdf_images"] == 1
        assert len(result.output["extracted_images"]) == 1
        assert result.output["observed"]["dimensions"]["width"] == 64
        assert "PDF Embedded Image Analysis" in result.output["analysis"]
    finally:
        db.query(DocumentRecord).filter(DocumentRecord.id == doc_id).delete()
        db.commit()
        db.close()
        if file_path.exists():
            file_path.unlink()


def test_analyze_image_tool_on_pdf_without_images(synthetic_blank_pdf):
    """Verifies analyze_image tool execution on a PDF with no raster images returns safe NO_IMAGES_FOUND."""
    db = SessionLocal()
    doc_id = f"doc_blank_{uuid.uuid4().hex[:8]}"
    file_path = DOCUMENTS_DIR / f"{doc_id}.pdf"
    file_path.write_bytes(synthetic_blank_pdf)

    try:
        rec = DocumentRecord(
            id=doc_id,
            filename="text_only.pdf",
            file_type="pdf",
            file_size_bytes=len(synthetic_blank_pdf),
            sha256_hash=uuid.uuid4().hex,
            mime_type="application/pdf",
            local_path=str(file_path),
        )
        db.add(rec)
        db.commit()

        tool = tool_registry.get_tool("analyze_image")
        result = tool.execute({"document_id": doc_id})
        assert result.success is True
        assert result.output["status"] == "NO_IMAGES_FOUND"
        assert result.output["extracted_images"] == []
        assert "No embedded raster images found" in result.output["analysis"]
    finally:
        db.query(DocumentRecord).filter(DocumentRecord.id == doc_id).delete()
        db.commit()
        db.close()
        if file_path.exists():
            file_path.unlink()


def test_analyze_image_tool_rejection_of_non_image_document():
    """Verifies that analyze_image tool safely rejects non-image, non-pdf documents."""
    db = SessionLocal()
    doc_id = f"doc_txt_{uuid.uuid4().hex[:8]}"
    file_path = DOCUMENTS_DIR / f"{doc_id}.txt"
    file_path.write_text("Plain text document.")

    try:
        rec = DocumentRecord(
            id=doc_id,
            filename="notes.txt",
            file_type="txt",
            file_size_bytes=21,
            sha256_hash=uuid.uuid4().hex,
            mime_type="text/plain",
            local_path=str(file_path),
        )
        db.add(rec)
        db.commit()

        tool = tool_registry.get_tool("analyze_image")
        result = tool.execute({"document_id": doc_id})
        assert result.success is False
        assert "is not an image or PDF" in result.error
    finally:
        db.query(DocumentRecord).filter(DocumentRecord.id == doc_id).delete()
        db.commit()
        db.close()
        if file_path.exists():
            file_path.unlink()


# =====================================================================
# 6. Zero Unrestricted Camera Monitoring
# =====================================================================

def test_zero_unrestricted_camera_monitoring():
    """
    Verifies privacy constraint: Zero camera surveillance, no OpenCV VideoCapture loops,
    no background frame-grabbing threads, and no continuous webcam ingestion daemons.
    Vision analysis operates strictly on explicit local file payloads.
    """
    import inspect
    import backend.models_local.clip_vision as cv_mod
    import backend.routes_vision as rv_mod

    # Verify no opencv / cv2 imports or video capture devices
    src_vision = inspect.getsource(cv_mod)
    src_routes = inspect.getsource(rv_mod)

    assert "VideoCapture" not in src_vision
    assert "cv2" not in src_vision
    assert "webcam" not in src_vision.lower()
    assert "VideoCapture" not in src_routes
    assert "webcam" not in src_routes.lower()

    # Verify absence of active camera polling background threads
    import threading
    active_thread_names = [t.name for t in threading.enumerate()]
    for name in active_thread_names:
        assert "camera" not in name.lower()
        assert "webcam" not in name.lower()
        assert "video_stream" not in name.lower()


# =====================================================================
# 7. REST API Endpoints
# =====================================================================

def test_api_vision_status_endpoint(client):
    """Verifies GET /vision/status returns model status and NPU feasibility."""
    response = client.get("/vision/status")
    assert response.status_code == 200
    data = response.json()
    assert "ResNet" in data["model_name"]
    assert data["target_hardware"] == "Snapdragon X Elite Hexagon NPU"
    assert "observed" in data["epistemic_separation"]
    assert "inferred" in data["epistemic_separation"]
    assert "jpg" in data["supported_formats"]


def test_api_vision_analyze_png_endpoint(client, synthetic_png_bytes):
    """Verifies POST /vision/analyze successfully processes an uploaded PNG."""
    response = client.post(
        "/vision/analyze",
        data={"prompt": "Identify primary colors and structure"},
        files={"file": ("chart.png", synthetic_png_bytes, "image/png")},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "VERIFIED"
    assert data["observed"]["dimensions"]["width"] == 120
    assert data["observed"]["dimensions"]["height"] == 80
    assert data["observed"]["format"] == "PNG"
    assert "visual_category" in data["inferred"]
    assert "capabilities_boundary" in data["inferred"]


def test_api_vision_analyze_pdf_endpoint(client, synthetic_pdf_with_image):
    """Verifies POST /vision/analyze successfully extracts and analyzes embedded images from PDF."""
    response = client.post(
        "/vision/analyze",
        data={"prompt": "Inspect schematic in PDF"},
        files={"file": ("spec.pdf", synthetic_pdf_with_image, "application/pdf")},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "VERIFIED"
    assert len(data["extracted_images"]) == 1
    assert data["extracted_images"][0]["dimensions"]["width"] == 64
    assert data["observed"]["dimensions"]["width"] == 64


def test_api_vision_analyze_document_endpoint(client, synthetic_png_bytes):
    """Verifies POST /vision/analyze-document inspects an ingested document by ID."""
    db = SessionLocal()
    doc_id = f"doc_api_{uuid.uuid4().hex[:8]}"
    file_path = DOCUMENTS_DIR / f"{doc_id}.png"
    file_path.write_bytes(synthetic_png_bytes)

    try:
        rec = DocumentRecord(
            id=doc_id,
            filename="system_architecture.png",
            file_type="png",
            file_size_bytes=len(synthetic_png_bytes),
            sha256_hash=uuid.uuid4().hex,
            mime_type="image/png",
            local_path=str(file_path),
        )
        db.add(rec)
        db.commit()

        response = client.post(
            "/vision/analyze-document",
            json={"document_id": doc_id, "prompt": "Check component blocks"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["document_id"] == doc_id
        assert data["status"] == "VERIFIED"
        assert data["observed"]["dimensions"]["width"] == 120
    finally:
        db.query(DocumentRecord).filter(DocumentRecord.id == doc_id).delete()
        db.commit()
        db.close()
        if file_path.exists():
            file_path.unlink()


def test_api_vision_analyze_unsupported_file_rejected(client):
    """Verifies POST /vision/analyze rejects unsupported file extensions."""
    response = client.post(
        "/vision/analyze",
        files={"file": ("malicious.exe", b"MZ\x90\x00\x03\x00\x00\x00", "application/octet-stream")},
    )
    assert response.status_code == 400
    assert "Unsupported file format" in response.json()["detail"]


def test_api_vision_analyze_empty_file_rejected(client):
    """Verifies POST /vision/analyze rejects zero-byte uploads."""
    response = client.post(
        "/vision/analyze",
        files={"file": ("empty.png", b"", "image/png")},
    )
    assert response.status_code == 400
    assert "Uploaded file is empty" in response.json()["detail"]


# =====================================================================
# 8. Phase 18 Real Neural Inference Verification Tests
# =====================================================================

def test_model_file_missing_fails_cleanly(tmp_path):
    """Verifies that an absent model file fails gracefully without pretending to work."""
    from backend.interfaces.base import ModelStatus
    from backend.models_local.clip_vision import LocalClipVisionModel

    missing = tmp_path / "nonexistent_model.onnx"
    model = LocalClipVisionModel(model_path=missing)
    ok = model.load()
    assert ok is False
    assert model.status == ModelStatus.FAILED
    assert model.is_loaded is False

    with pytest.raises(RuntimeError, match="failed to load"):
        model.encode_image(b"some_bytes")


def test_valid_image_successful_real_inference(synthetic_png_bytes):
    """Verifies real neural inference runs, producing top candidate classes and timing metrics."""
    local_clip_vision.load()
    res = local_clip_vision.inspect_image(synthetic_png_bytes)

    assert "observed" in res
    assert "inferred" in res
    inferred = res["inferred"]

    assert inferred["confidence"] > 0.0
    assert len(inferred["top_candidates"]) == 5

    for cand in inferred["top_candidates"]:
        assert "label" in cand
        assert "confidence" in cand
        assert "class_index" in cand
        assert isinstance(cand["label"], str) and len(cand["label"]) > 0
        assert 0.0 <= cand["confidence"] <= 1.0

    assert res["preprocessing_ms"] >= 0.0
    assert res["inference_ms"] > 0.0
    assert res["postprocessing_ms"] >= 0.0
    assert res["duration_ms"] > 0.0


def test_malformed_image_raises_controlled_error():
    """Verifies that corrupted image bytes raise a controlled ValueError instead of crashing."""
    corrupted_data = b"NOT_A_VALID_IMAGE_FORMAT_DATA_STREAM" * 10
    with pytest.raises(ValueError):
        local_clip_vision.inspect_image(corrupted_data)

    with pytest.raises(ValueError):
        local_clip_vision.encode_image(corrupted_data)


def test_two_different_images_produce_different_outputs(synthetic_png_bytes, synthetic_jpg_bytes):
    """
    Core correctness test: Two distinct images must produce distinct neural embeddings
    and distinct classification distributions.
    """
    local_clip_vision.load()

    emb1 = local_clip_vision.encode_image(synthetic_png_bytes)
    emb2 = local_clip_vision.encode_image(synthetic_jpg_bytes)

    # 512-dim visual embeddings must differ between distinct visual signals
    assert emb1 != emb2
    # Dot product / cosine similarity of distinct synthetic images must be less than 1.0
    cos_sim = float(np.dot(emb1, emb2))
    assert cos_sim < 0.999

    res1 = local_clip_vision.inspect_image(synthetic_png_bytes)
    res2 = local_clip_vision.inspect_image(synthetic_jpg_bytes)

    # Inferred neural predictions must differ
    labels1 = [c["label"] for c in res1["inferred"]["top_candidates"]]
    labels2 = [c["label"] for c in res2["inferred"]["top_candidates"]]
    assert labels1 != labels2 or res1["inferred"]["confidence"] != res2["inferred"]["confidence"]


def test_cpu_provider_runs_real_inference(synthetic_png_bytes):
    """Verifies that CPU execution provider executes genuine ONNX model inference."""
    local_clip_vision.load()
    assert local_clip_vision.execution_provider in ["CPUExecutionProvider", "QNNExecutionProvider"]

    res = local_clip_vision.inspect_image(synthetic_png_bytes)
    assert res["execution_provider"] == local_clip_vision.execution_provider

    confs = [c["confidence"] for c in res["inferred"]["top_candidates"]]
    # Top candidates must be strictly sorted by descending probability
    assert confs == sorted(confs, reverse=True)


def test_qnn_unavailable_reports_honest_cpu_status():
    """Verifies that running on non-Snapdragon host does not falsely claim active NPU."""
    import platform
    if platform.machine().lower() not in ["arm64", "aarch64"]:
        local_clip_vision.load()
        status = local_clip_vision.get_status()
        assert status["execution_provider"] == "CPUExecutionProvider"
        assert "Hexagon NPU" not in status["current_hardware"]


def test_returned_embedding_has_expected_shape_and_norm(synthetic_png_bytes):
    """Verifies that encode_image produces a valid 512-dim L2-normalized float vector."""
    local_clip_vision.load()
    emb = local_clip_vision.encode_image(synthetic_png_bytes)

    assert len(emb) == 512
    assert all(isinstance(x, float) for x in emb)
    # L2 norm must be approximately 1.0
    norm = float(np.linalg.norm(emb))
    assert np.isclose(norm, 1.0, atol=1e-4)


def test_no_hardcoded_vision_heuristic_assignments():
    """Structural test: ensures fake threshold rules and hardcoded labels are completely removed."""
    import pathlib
    source_file = pathlib.Path(__file__).parent.parent / "backend" / "models_local" / "clip_vision.py"
    source_code = source_file.read_text(encoding="utf-8")

    # The old fake heuristic threshold assignments must be gone
    assert 'visual_category = "document_page"' not in source_code
    assert 'visual_category = "technical_diagram"' not in source_code
    assert 'visual_category = "dark_mode_ui_or_dashboard"' not in source_code
    assert "bins = np.linspace(0, len(flat), 513, dtype=int)" not in source_code
