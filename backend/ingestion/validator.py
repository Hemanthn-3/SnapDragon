import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from backend.errors import NexusException

MAX_FILE_SIZE_BYTES = 50 * 1024 * 1024  # 50 MB

SUPPORTED_EXTENSIONS = {
    ".pdf": "application/pdf",
    ".txt": "text/plain",
    ".json": "application/json",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
}

# Signatures for magic bytes validation
MAGIC_SIGNATURES = {
    ".pdf": [b"%PDF-"],
    ".docx": [b"PK\x03\x04"],
    ".png": [b"\x89PNG\r\n\x1a\n"],
    ".jpg": [b"\xff\xd8\xff"],
    ".jpeg": [b"\xff\xd8\xff"],
}


class DocumentValidationError(NexusException):
    """Raised when an uploaded document fails format, magic byte, or size validation."""
    def __init__(self, detail: str):
        super().__init__(message=detail, status_code=400)


@dataclass
class ValidationResult:
    """Metadata extracted during file validation."""
    filename: str
    extension: str
    file_type: str
    mime_type: str
    file_size_bytes: int
    sha256_hash: str


WINDOWS_RESERVED_NAMES = {
    "CON", "PRN", "AUX", "NUL",
    "COM1", "COM2", "COM3", "COM4", "COM5", "COM6", "COM7", "COM8", "COM9",
    "LPT1", "LPT2", "LPT3", "LPT4", "LPT5", "LPT6", "LPT7", "LPT8", "LPT9",
}


class DocumentValidator:
    """Validates uploaded files to ensure integrity, offline compatibility, and security."""

    @staticmethod
    def validate(filename: str, content: bytes) -> ValidationResult:
        if not filename or not filename.strip():
            raise DocumentValidationError("Filename must not be empty.")

        # 1. Reject forbidden null bytes
        if "\x00" in filename:
            raise DocumentValidationError("Filename contains forbidden null byte.")

        # 2. Reject path traversal characters
        if ".." in filename or "/" in filename or "\\" in filename:
            raise DocumentValidationError(f"Filename '{filename}' contains invalid path traversal characters.")

        # 3. Enforce maximum filename length
        if len(filename) > 255:
            raise DocumentValidationError("Filename exceeds maximum allowed length of 255 characters.")

        # 4. Check Windows reserved device names
        stem = Path(filename).stem.upper()
        if stem in WINDOWS_RESERVED_NAMES:
            raise DocumentValidationError(f"Filename '{filename}' uses a reserved system device name.")

        ext = Path(filename).suffix.lower()
        if ext not in SUPPORTED_EXTENSIONS:
            allowed = ", ".join(sorted(SUPPORTED_EXTENSIONS.keys()))
            raise DocumentValidationError(
                f"Unsupported file format '{ext}'. Supported formats: {allowed}."
            )

        file_size = len(content)
        if file_size == 0:
            raise DocumentValidationError("File is empty (0 bytes).")

        if file_size > MAX_FILE_SIZE_BYTES:
            raise DocumentValidationError(
                f"File size ({file_size / (1024 * 1024):.1f}MB) exceeds maximum limit of {MAX_FILE_SIZE_BYTES / (1024 * 1024):.0f}MB."
            )

        # Magic byte signature check
        if ext in MAGIC_SIGNATURES:
            valid_sig = any(content.startswith(sig) for sig in MAGIC_SIGNATURES[ext])
            if not valid_sig:
                raise DocumentValidationError(
                    f"File magic signature mismatch: '{filename}' does not appear to be a valid {ext.upper()} file."
                )
        elif ext == ".txt":
            # Verify text is decodable as utf-8 or ascii
            try:
                content.decode("utf-8")
            except UnicodeDecodeError:
                try:
                    content.decode("latin-1")
                except Exception:
                    raise DocumentValidationError(
                        f"Text file '{filename}' contains invalid or non-decodable binary data."
                    )

        # Calculate SHA256 checksum
        sha256 = hashlib.sha256(content).hexdigest()
        file_type = ext.lstrip(".")
        mime_type = SUPPORTED_EXTENSIONS[ext]

        return ValidationResult(
            filename=filename,
            extension=ext,
            file_type=file_type,
            mime_type=mime_type,
            file_size_bytes=file_size,
            sha256_hash=sha256,
        )
