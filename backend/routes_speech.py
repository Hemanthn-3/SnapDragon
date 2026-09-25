"""
NEXUS Speech Recognition API Routes (Phase 17 — Real ASR)
Endpoints for local Whisper-based transcription. Zero cloud calls.
"""

from typing import Any, Dict, List, Optional
from fastapi import APIRouter, File, HTTPException, UploadFile, status
from pydantic import BaseModel, Field

from backend.logger import get_logger
from backend.models_local.whisper_speech import local_speech_model

logger = get_logger("nexus.routes_speech")

router = APIRouter(prefix="/speech", tags=["Local Speech Recognition (ASR)"])

# Maximum file size accepted for audio uploads (10 MB)
MAX_AUDIO_BYTES = 10 * 1024 * 1024


class SpeechStatusResponse(BaseModel):
    model_name: str
    target_hardware: str
    execution_provider: str
    runtime: str
    status: str
    is_loaded: bool
    blockers: List[str]


class TranscribeResponse(BaseModel):
    transcript: str = Field(..., description="Recognized speech text (empty string for silence)")
    language: str = Field(default="en")
    duration_seconds: float
    energy: float
    silence: bool = Field(default=False)
    hardware: str
    runtime: str
    execution_provider: str
    latency_ms: float
    success: bool
    blockers: List[str] = Field(default_factory=list)


@router.get(
    "/status",
    response_model=SpeechStatusResponse,
    summary="Get ASR Model Status & Hardware Diagnostics",
    description=(
        "Returns the lifecycle status, active runtime, execution provider, "
        "and Snapdragon compatibility blockers if any."
    ),
)
def get_speech_status():
    health = local_speech_model.health_check()
    return SpeechStatusResponse(
        model_name=health["model"],
        target_hardware=health["target_hardware"],
        execution_provider=health["execution_provider"],
        runtime=health["runtime"],
        status=health["status"],
        is_loaded=health["is_loaded"],
        blockers=health["blockers"],
    )


@router.post(
    "/transcribe",
    response_model=TranscribeResponse,
    summary="Transcribe Audio Locally",
    description=(
        "Decodes audio bytes and transcribes speech using real local Whisper inference. "
        "Zero cloud API calls. Returns actual model output — no hard-coded transcripts."
    ),
)
async def transcribe_audio(file: UploadFile = File(...)):
    if not file:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="No audio file provided.",
        )

    logger.info(
        f"[ASR_API] Received audio: filename='{file.filename}', "
        f"content_type='{file.content_type}'"
    )
    audio_bytes = await file.read()

    # Reject empty or undersized payload (less than a minimal WAV header)
    if not audio_bytes or len(audio_bytes) < 44:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Audio payload is empty or too short to contain a valid audio stream.",
        )

    # Reject oversized files
    if len(audio_bytes) > MAX_AUDIO_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"Audio file exceeds maximum size of {MAX_AUDIO_BYTES // 1024 // 1024} MB.",
        )

    try:
        result = local_speech_model.transcribe(audio_bytes=audio_bytes)

        # If model returned an explicit failure dict (not an exception)
        if not result.get("success", True):
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=result.get("error", "Speech recognition inference failed."),
            )

        return TranscribeResponse(
            transcript=result.get("text", ""),
            language=result.get("language", "en"),
            duration_seconds=result.get("duration_seconds", 0.0),
            energy=result.get("energy", 0.0),
            silence=result.get("silence", False),
            hardware=result.get("hardware", local_speech_model.target_hardware),
            runtime=result.get("runtime", local_speech_model.runtime),
            execution_provider=result.get(
                "execution_provider", local_speech_model.execution_provider
            ),
            latency_ms=result.get("latency_ms", 0.0),
            success=True,
            blockers=result.get("blockers", []),
        )

    except HTTPException:
        raise  # Re-raise structured HTTP errors

    except ValueError as ve:
        logger.warning(f"[ASR_API] Invalid audio format: {ve}")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(ve),
        )
    except Exception as e:
        logger.error(f"[ASR_API] Transcription error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Local ASR transcription error: {str(e)}",
        )
