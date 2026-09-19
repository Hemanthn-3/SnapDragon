import hashlib
from pathlib import Path
from fastapi import status

from tests.fixtures_data import (
    create_synthetic_pdf,
    create_scanned_synthetic_pdf,
    create_synthetic_docx,
    create_synthetic_txt,
    create_synthetic_png,
    create_synthetic_jpg,
    create_invalid_file,
)


def test_pdf_extraction_and_page_numbers(client):
    """Verifies that PDF text extraction extracts text page-by-page preserving 1-indexed page numbers."""
    pdf_bytes = create_synthetic_pdf(num_pages=3)
    files = {"file": ("report.pdf", pdf_bytes, "application/pdf")}

    response = client.post("/documents", files=files)
    assert response.status_code == status.HTTP_201_CREATED

    data = response.json()["document"]
    doc_id = data["id"]
    assert data["filename"] == "report.pdf"
    assert data["file_type"] == "pdf"
    assert data["page_count"] == 3
    assert data["chunk_count"] >= 3
    assert data["ocr_status"] == "NOT_APPLICABLE"

    # Verify every chunk preserves required fields
    pages_seen = set()
    for chunk in data["chunks"]:
        assert chunk["document_id"] == doc_id
        assert chunk["filename"] == "report.pdf"
        assert chunk["chunk_id"] is not None
        assert chunk["source_location"].startswith("page:")
        assert chunk["page_number"] in [1, 2, 3]
        assert chunk["char_count"] > 0
        assert chunk["word_count"] > 0
        pages_seen.add(chunk["page_number"])

    assert pages_seen == {1, 2, 3}

    # Cleanup
    client.delete(f"/documents/{doc_id}")


def test_docx_extraction_paragraphs_and_tables(client):
    """Verifies that DOCX files extract paragraph text and table content with source locations."""
    docx_bytes = create_synthetic_docx()
    files = {"file": ("architecture.docx", docx_bytes, "application/vnd.openxmlformats-officedocument.wordprocessingml.document")}

    response = client.post("/documents", files=files)
    assert response.status_code == status.HTTP_201_CREATED

    data = response.json()["document"]
    doc_id = data["id"]
    assert data["file_type"] == "docx"
    assert data["chunk_count"] >= 3

    # Verify paragraph and table extraction
    chunk_texts = [c["text"] for c in data["chunks"]]
    source_locs = [c["source_location"] for c in data["chunks"]]

    assert any("NEXUS is an offline-first multimodal AI work agent" in t for t in chunk_texts)
    assert any("Whisper-Small" in t for t in chunk_texts)
    assert any("table:" in loc for loc in source_locs)

    # Cleanup
    client.delete(f"/documents/{doc_id}")


def test_txt_extraction(client):
    """Verifies that plain text files extract paragraph chunks."""
    txt_bytes = create_synthetic_txt()
    files = {"file": ("notes.txt", txt_bytes, "text/plain")}

    response = client.post("/documents", files=files)
    assert response.status_code == status.HTTP_201_CREATED

    data = response.json()["document"]
    doc_id = data["id"]
    assert data["file_type"] == "txt"
    assert data["chunk_count"] >= 2

    # Cleanup
    client.delete(f"/documents/{doc_id}")


def test_image_handling_ocr_not_yet_implemented(client):
    """Verifies that images are handled, validated, and report OCR STATUS: NOT YET IMPLEMENTED without faking text."""
    png_bytes = create_synthetic_png(width=300, height=200)
    files = {"file": ("screenshot.png", png_bytes, "image/png")}

    response = client.post("/documents", files=files)
    assert response.status_code == status.HTTP_201_CREATED

    data = response.json()["document"]
    doc_id = data["id"]
    assert data["file_type"] == "png"
    assert data["ocr_status"] == "NOT_YET_IMPLEMENTED"
    # No faked OCR chunks
    assert data["chunk_count"] == 0

    # Test JPEG image
    jpg_bytes = create_synthetic_jpg(width=150, height=150)
    jpg_files = {"file": ("photo.jpg", jpg_bytes, "image/jpeg")}
    jpg_resp = client.post("/documents", files=jpg_files)
    assert jpg_resp.status_code == status.HTTP_201_CREATED
    jpg_data = jpg_resp.json()["document"]
    assert jpg_data["ocr_status"] == "NOT_YET_IMPLEMENTED"

    # Cleanup
    client.delete(f"/documents/{doc_id}")
    client.delete(f"/documents/{jpg_data['id']}")


def test_scanned_pdf_ocr_status(client):
    """Verifies that scanned PDFs with no extractable digital text report OCR STATUS: NOT YET IMPLEMENTED."""
    scanned_pdf = create_scanned_synthetic_pdf()
    files = {"file": ("scanned_invoice.pdf", scanned_pdf, "application/pdf")}

    response = client.post("/documents", files=files)
    assert response.status_code == status.HTTP_201_CREATED

    data = response.json()["document"]
    doc_id = data["id"]
    assert data["page_count"] == 1
    assert data["ocr_status"] == "NOT_YET_IMPLEMENTED"
    assert data["chunk_count"] == 0

    # Cleanup
    client.delete(f"/documents/{doc_id}")


def test_invalid_files_rejection(client):
    """Verifies that invalid extensions, corrupted signatures, and empty files are rejected."""
    # 1. Unsupported extension (.exe)
    bad_ext_files = {"file": ("malware.exe", b"MZ\x90\x00", "application/octet-stream")}
    resp = client.post("/documents", files=bad_ext_files)
    assert resp.status_code == status.HTTP_400_BAD_REQUEST
    assert "Unsupported file format" in resp.json()["message"]

    # 2. Corrupted signature (.pdf with non-PDF content)
    invalid_bytes = create_invalid_file()
    fake_pdf_files = {"file": ("corrupt.pdf", invalid_bytes, "application/pdf")}
    resp = client.post("/documents", files=fake_pdf_files)
    assert resp.status_code == status.HTTP_400_BAD_REQUEST
    assert "magic signature mismatch" in resp.json()["message"]

    # 3. Empty file (0 bytes)
    empty_files = {"file": ("empty.txt", b"", "text/plain")}
    resp = client.post("/documents", files=empty_files)
    assert resp.status_code == status.HTTP_400_BAD_REQUEST
    assert "empty" in resp.json()["message"]


def test_metadata_extraction_and_hash(client):
    """Verifies exact metadata extraction: size, SHA256 checksum, and MIME types."""
    content = b"Exact deterministic content for SHA256 checksum test."
    expected_hash = hashlib.sha256(content).hexdigest()
    files = {"file": ("test_hash.txt", content, "text/plain")}

    response = client.post("/documents", files=files)
    assert response.status_code == status.HTTP_201_CREATED

    data = response.json()["document"]
    doc_id = data["id"]
    assert data["sha256_hash"] == expected_hash
    assert data["file_size_bytes"] == len(content)
    assert data["mime_type"] == "text/plain"

    # Cleanup
    client.delete(f"/documents/{doc_id}")


def test_local_storage_and_file_deletion(client):
    """Verifies files are stored strictly locally and completely purged on document deletion."""
    content = create_synthetic_pdf(num_pages=1)
    files = {"file": ("local_test.pdf", content, "application/pdf")}

    # Ingest
    response = client.post("/documents", files=files)
    assert response.status_code == status.HTTP_201_CREATED
    data = response.json()["document"]
    doc_id = data["id"]

    # Verify physical file existence on disk
    expected_dir = Path("data/documents") / doc_id
    expected_file = expected_dir / "local_test.pdf"
    assert expected_dir.exists()
    assert expected_file.exists()
    assert expected_file.read_bytes() == content

    # Delete
    del_resp = client.delete(f"/documents/{doc_id}")
    assert del_resp.status_code == status.HTTP_200_OK

    # Verify physical file removal from disk
    assert not expected_dir.exists()
    assert not expected_file.exists()

    # Verify DB record is gone (404)
    get_resp = client.get(f"/documents/{doc_id}")
    assert get_resp.status_code == status.HTTP_404_NOT_FOUND


def test_document_api_crud_workflow(client):
    """Tests full end-to-end CRUD workflow: POST, GET, GET/{id}, DELETE."""
    txt_content = b"CRUD workflow testing text."
    files = {"file": ("crud.txt", txt_content, "text/plain")}

    # 1. Create
    create_resp = client.post("/documents", files=files)
    assert create_resp.status_code == status.HTTP_201_CREATED
    doc_id = create_resp.json()["document"]["id"]

    # 2. List
    list_resp = client.get("/documents")
    assert list_resp.status_code == status.HTTP_200_OK
    doc_list = list_resp.json()["documents"]
    assert any(d["id"] == doc_id for d in doc_list)

    # 3. Retrieve detail with chunks
    detail_resp = client.get(f"/documents/{doc_id}")
    assert detail_resp.status_code == status.HTTP_200_OK
    assert detail_resp.json()["document"]["id"] == doc_id
    assert len(detail_resp.json()["document"]["chunks"]) > 0

    # 4. Delete
    del_resp = client.delete(f"/documents/{doc_id}")
    assert del_resp.status_code == status.HTTP_200_OK

    # 5. Confirm 404 after deletion
    post_del_resp = client.get(f"/documents/{doc_id}")
    assert post_del_resp.status_code == status.HTTP_404_NOT_FOUND
