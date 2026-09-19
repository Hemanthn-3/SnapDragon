"""
NEXUS Phase 8: Speech Recognition API Routes
Provides endpoints for inspecting ASR engine status, Snapdragon blockers,
and transcribing voice input locally without external cloud dependencies.
"""

from typing import Any, Dict, List
from fastapi import APIRouter, File, HTTPException, UploadFile, status
from pydantic import BaseModel, Field

from backend.logger import get_logger
from backend.models_local.whisper_speech import local_speech_model

logger = get_logger("nexus.routes_speech")

router = APIRouter(prefix="/speech", tags=["Local Speech Recognition (ASR)"])


class SpeechStatusResponse(BaseModel):
    model_name: str
    target_hardware: str
    execution_provider: str
    status: str
    is_loaded: bool
    blockers: List[str]


class TranscribeResponse(BaseModel):
    transcript: str = Field(..., description="Recognized speech text")
    language: str = Field(default="en")
    duration_seconds: float
    hardware: str
    execution_provider: str
    blockers: List[str] = Field(default_factory=list)


@router.get(
    "/status",
    response_model=SpeechStatusResponse,
    summary="Get Speech Model Status & Hardware Diagnostics",
    description="Returns lifecycle status, active execution provider, and Snapdragon compatibility blockers if any.",
)
def get_speech_status():
    metadata = local_speech_model.get_metadata()
    return SpeechStatusResponse(
        model_name=metadata["model_name"],
        target_hardware=local_speech_model.target_hardware,
        execution_provider=local_speech_model.execution_provider,
        status=metadata["status"],
        is_loaded=metadata["is_loaded"],
        blockers=local_speech_model.blockers,
    )


@router.post(
    "/transcribe",
    response_model=TranscribeResponse,
    summary="Transcribe Local Audio Payload",
    description="Decodes local audio bytes and transcribes speech using Whisper-Small-Quantized. Zero cloud API calls.",
)
async def transcribe_audio(file: UploadFile = File(...)):
    if not file:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="No audio file provided.",
        )

    logger.info(f"[ASR_API] Received audio upload: filename='{file.filename}', content_type='{file.content_type}'")
    audio_bytes = await file.read()

    if not audio_bytes or len(audio_bytes) < 44:  # Minimum valid WAV header size
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Audio payload is empty or too short to contain a valid audio stream.",
        )

    try:
        result = local_speech_model.transcribe(audio_bytes=audio_bytes)
        return TranscribeResponse(
            transcript=result["text"],
            language=result["language"],
            duration_seconds=result["duration_seconds"],
            hardware=result["hardware"],
            execution_provider=result["execution_provider"],
            blockers=result["blockers"],
        )
    except ValueError as ve:
        logger.warning(f"[ASR_API] Invalid audio format: {ve}")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(ve),
        )
    except Exception as e:
        logger.error(f"[ASR_API] Transcription failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Local ASR transcription error: {str(e)}",
        )
