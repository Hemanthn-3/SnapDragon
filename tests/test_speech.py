"""
NEXUS Phase 8: Local Speech Recognition (ASR) Tests
Tests Whisper-Small-Quantized model adapter, Snapdragon blocker detection,
audio decoding, and REST endpoints with zero cloud calls.
"""

import io
import wave
import pytest
import numpy as np

from backend.interfaces.base import ModelStatus
from backend.models_local.whisper_speech import LocalWhisperSpeechModel, local_speech_model


def create_synthetic_wav(duration_seconds: float = 1.0, sample_rate: int = 16000, frequency: float = 440.0) -> bytes:
    """Generates an in-memory 16kHz mono 16-bit PCM RIFF WAV audio file."""
    n_samples = int(sample_rate * duration_seconds)
    t = np.linspace(0, duration_seconds, n_samples, endpoint=False)
    # 440 Hz sine tone with amplitude
    samples = (np.sin(2 * np.pi * frequency * t) * 16000).astype(np.int16)

    bio = io.BytesIO()
    with wave.open(bio, "wb") as wf:
        wf.setnchannels(1)        # mono
        wf.setsampwidth(2)        # 16-bit
        wf.setframerate(sample_rate)
        wf.writeframes(samples.tobytes())
    return bio.getvalue()


def create_silent_wav(duration_seconds: float = 1.0, sample_rate: int = 16000) -> bytes:
    """Generates an in-memory silent 16kHz mono WAV file."""
    n_samples = int(sample_rate * duration_seconds)
    samples = np.zeros(n_samples, dtype=np.int16)

    bio = io.BytesIO()
    with wave.open(bio, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(samples.tobytes())
    return bio.getvalue()


# =====================================================================
# 1. Test Model Metadata & Hardware Diagnostics
# =====================================================================

def test_speech_model_status_and_metadata():
    model = LocalWhisperSpeechModel()
    meta = model.get_metadata()

    assert meta["model_name"] == "Whisper-Small-Quantized"
    assert "target_hardware" in meta
    assert model.status in [ModelStatus.NOT_LOADED, ModelStatus.READY]

    # Test load
    assert model.load() is True
    assert model.status == ModelStatus.READY
    assert model.is_loaded is True
    assert model.execution_provider in ["QNNExecutionProvider", "CPUExecutionProvider"]


def test_hardware_blocker_detection_on_development_host():
    """
    CRITICAL RULE: If the model cannot yet run on the target Snapdragon environment,
    document the exact blocker instead of creating a fake implementation.
    """
    import platform
    model = LocalWhisperSpeechModel()
    machine = platform.machine().upper()

    if machine not in ["ARM64", "AARCH64"]:
        # On x86_64 host, blockers must be explicitly documented
        blockers = model.blockers
        assert len(blockers) >= 1
        assert any("Architecture Mismatch" in b or "Hexagon NPU" in b for b in blockers)
        assert any("QNN" in b or "QnnHtp.dll" in b for b in blockers)
        # Target hardware must not claim Hexagon NPU on AMD64
        assert "CPU" in model.target_hardware
        assert model.execution_provider != "QNNExecutionProvider"


# =====================================================================
# 2. Test Audio Decoding & Local Transcription Contract
# =====================================================================

def test_audio_transcription_synthetic_wav():
    model = LocalWhisperSpeechModel()
    wav_bytes = create_synthetic_wav(duration_seconds=1.5, sample_rate=16000)

    result = model.transcribe(audio_bytes=wav_bytes, sample_rate=16000)

    assert "text" in result
    assert isinstance(result["text"], str)
    assert len(result["text"]) > 0
    assert result["language"] == "en"
    assert 1.4 <= result["duration_seconds"] <= 1.6
    assert result["sample_rate"] == 16000
    assert result["energy"] > 0.001
    assert result["model"] == "Whisper-Small-Quantized"
    assert "hardware" in result
    assert "execution_provider" in result
    assert "blockers" in result


def test_audio_silence_detection():
    model = LocalWhisperSpeechModel()
    silent_bytes = create_silent_wav(duration_seconds=1.0, sample_rate=16000)

    result = model.transcribe(audio_bytes=silent_bytes)

    assert result["energy"] < 0.001
    assert "[Silence" in result["text"]


def test_empty_audio_rejected():
    model = LocalWhisperSpeechModel()
    with pytest.raises(ValueError) as exc_info:
        model.transcribe(audio_bytes=b"")
    assert "Audio payload is empty" in str(exc_info.value)


# =====================================================================
# 3. Test REST API Endpoints
# =====================================================================

def test_api_speech_status_endpoint(client):
    response = client.get("/speech/status")
    assert response.status_code == 200
    data = response.json()

    assert data["model_name"] == "Whisper-Small-Quantized"
    assert "target_hardware" in data
    assert "execution_provider" in data
    assert "status" in data
    assert isinstance(data["blockers"], list)


def test_api_speech_transcribe_endpoint(client):
    wav_bytes = create_synthetic_wav(duration_seconds=1.2)
    files = {"file": ("test_input.wav", io.BytesIO(wav_bytes), "audio/wav")}

    response = client.post("/speech/transcribe", files=files)
    assert response.status_code == 200
    data = response.json()

    assert "transcript" in data
    assert len(data["transcript"]) > 0
    assert data["language"] == "en"
    assert 1.1 <= data["duration_seconds"] <= 1.3
    assert "hardware" in data
    assert "execution_provider" in data


def test_api_speech_transcribe_empty_audio_rejected(client):
    files = {"file": ("empty.wav", io.BytesIO(b""), "audio/wav")}
    response = client.post("/speech/transcribe", files=files)
    assert response.status_code == 422


# =====================================================================
# 4. Test No Auto-Execution Guarantee
# =====================================================================

def test_transcription_does_not_auto_execute_plan(client):
    """
    CRITICAL RULE:
    "Do not automatically execute a task immediately after transcription.
    Allow the user to edit the transcript."
    Verifies that calling /speech/transcribe produces only the text and does not
    trigger planner or executor endpoints.
    """
    wav_bytes = create_synthetic_wav(duration_seconds=1.0)
    files = {"file": ("goal_voice.wav", io.BytesIO(wav_bytes), "audio/wav")}

    response = client.post("/speech/transcribe", files=files)
    assert response.status_code == 200
    data = response.json()

    # Must contain only transcript payload
    assert "transcript" in data
    # Must NOT contain a plan, tasks, or execution output
    assert "plan" not in data
    assert "tasks" not in data
    assert "execution_result" not in data
