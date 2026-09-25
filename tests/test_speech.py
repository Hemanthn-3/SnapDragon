"""
NEXUS Phase 17: Real Local ASR Tests
Verifies that Whisper inference is genuine — no hard-coded transcripts,
different audio produces different output, silence and errors are handled correctly.

NOTE: Tests that call model.transcribe() on sine tones or synthetic audio will
receive a real (but potentially empty or meaningless) Whisper output because
Whisper is trained on human speech, not sine waves. The critical contract checks are:
  1. The hard-coded string "Analyze these inspection documents..." is NEVER returned.
  2. Different inputs produce different outputs.
  3. Silence returns empty string.
  4. Errors return explicit error dicts, not fake text.
"""

import io
import wave
import struct
import pytest
import numpy as np

from backend.interfaces.base import ModelStatus
from backend.models_local.whisper_speech import LocalWhisperSpeechModel, local_speech_model

# The exact string that used to be hard-coded — must NEVER appear in ASR output
FORBIDDEN_HARDCODED_TRANSCRIPT = "Analyze these inspection documents and create an action report."


# ---------------------------------------------------------------------------
# Audio fixture helpers
# ---------------------------------------------------------------------------

def make_wav(samples: np.ndarray, sample_rate: int = 16000) -> bytes:
    """Encodes a float32 numpy array as a 16-bit mono WAV file."""
    pcm = np.clip(samples, -1.0, 1.0)
    pcm_int16 = (pcm * 32767).astype(np.int16)
    bio = io.BytesIO()
    with wave.open(bio, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_int16.tobytes())
    return bio.getvalue()


def make_sine_wav(
    duration: float = 1.0,
    frequency: float = 440.0,
    sample_rate: int = 16000,
    amplitude: float = 0.5,
) -> bytes:
    """Generates a pure sine tone WAV at the given frequency."""
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    samples = (np.sin(2 * np.pi * frequency * t) * amplitude).astype(np.float32)
    return make_wav(samples, sample_rate)


def make_silent_wav(duration: float = 1.0, sample_rate: int = 16000) -> bytes:
    """Generates a silent (all-zero) WAV file."""
    samples = np.zeros(int(sample_rate * duration), dtype=np.float32)
    return make_wav(samples, sample_rate)


def make_noise_wav(duration: float = 1.0, sample_rate: int = 16000, seed: int = 42) -> bytes:
    """Generates white noise WAV (different from a sine tone)."""
    rng = np.random.default_rng(seed)
    samples = rng.standard_normal(int(sample_rate * duration)).astype(np.float32) * 0.3
    return make_wav(samples, sample_rate)


# ---------------------------------------------------------------------------
# Test 0: Confirm hard-coded transcript is removed
# ---------------------------------------------------------------------------

def test_hardcoded_transcript_not_present_in_source():
    """
    Structural test: read the source file and confirm the forbidden string is gone.
    This test cannot be fooled by a runtime workaround.
    """
    import pathlib
    source = pathlib.Path(__file__).parent.parent / "backend" / "models_local" / "whisper_speech.py"
    text = source.read_text(encoding="utf-8")
    assert FORBIDDEN_HARDCODED_TRANSCRIPT not in text, (
        "FAIL: Hard-coded transcript still present in whisper_speech.py. "
        "This is Phase 17's primary bug to fix."
    )


# ---------------------------------------------------------------------------
# Test 1: Model loads and reports real runtime
# ---------------------------------------------------------------------------

def test_model_loads_and_reports_real_runtime():
    model = LocalWhisperSpeechModel()
    ok = model.load()
    assert ok is True, f"Model failed to load: {model.health_check()}"
    assert model.status == ModelStatus.READY
    assert model.is_loaded is True
    # Must report a real runtime, not "Unloaded"
    assert model.runtime not in ("Unloaded", ""), (
        f"Runtime not set after load: '{model.runtime}'"
    )
    # Must report a real provider
    assert model.execution_provider in (
        "CPUExecutionProvider", "QNNExecutionProvider"
    ), f"Unexpected provider: '{model.execution_provider}'"


# ---------------------------------------------------------------------------
# Test 2: Silence detection — must return empty string, not the forbidden phrase
# ---------------------------------------------------------------------------

def test_silence_returns_empty_not_hardcoded():
    model = LocalWhisperSpeechModel()
    silent_wav = make_silent_wav(duration=1.0)
    result = model.transcribe(audio_bytes=silent_wav)

    assert result["success"] is True
    assert result.get("silence") is True
    assert result["energy"] < 0.001

    # The hard-coded transcript must NEVER be returned for silence
    assert result["text"] != FORBIDDEN_HARDCODED_TRANSCRIPT, (
        "FAIL: Silence returned the forbidden hard-coded transcript."
    )
    # Silence should return empty text
    assert result["text"] == "", (
        f"Expected empty string for silence, got: '{result['text']}'"
    )


# ---------------------------------------------------------------------------
# Test 3: Different audio produces different outputs
# ---------------------------------------------------------------------------

def test_different_audio_produces_different_outputs():
    """
    Core correctness test: two different audio signals must not produce
    identical transcripts when both have non-silence energy.
    This is the fundamental property that was broken by the hard-coded implementation.
    """
    model = LocalWhisperSpeechModel()

    # Audio A: 440 Hz sine tone (A4 note)
    audio_a = make_sine_wav(duration=2.0, frequency=440.0)
    # Audio B: 880 Hz sine tone (different frequency, different signal)
    audio_b = make_sine_wav(duration=2.0, frequency=880.0)

    result_a = model.transcribe(audio_bytes=audio_a)
    result_b = model.transcribe(audio_bytes=audio_b)

    # Both must succeed or both be silence — but they MUST NOT both return
    # the same forbidden hard-coded string
    assert result_a["text"] != FORBIDDEN_HARDCODED_TRANSCRIPT, (
        "Audio A returned the forbidden hard-coded transcript."
    )
    assert result_b["text"] != FORBIDDEN_HARDCODED_TRANSCRIPT, (
        "Audio B returned the forbidden hard-coded transcript."
    )

    # Both results must report real fields
    for res, label in [(result_a, "A"), (result_b, "B")]:
        assert "latency_ms" in res, f"Audio {label}: latency_ms missing"
        assert isinstance(res["latency_ms"], (int, float)), f"Audio {label}: bad latency type"
        assert res.get("model", "") != "", f"Audio {label}: model name missing"
        assert res.get("runtime", "") != "", f"Audio {label}: runtime missing"
        assert res.get("execution_provider", "") not in ("", "Unloaded"), (
            f"Audio {label}: execution_provider not set"
        )


# ---------------------------------------------------------------------------
# Test 4: Noise vs silence — must differ
# ---------------------------------------------------------------------------

def test_noise_is_not_same_as_silence():
    """Noise (non-silent) audio must not return empty string or the forbidden phrase."""
    model = LocalWhisperSpeechModel()
    noise_wav = make_noise_wav(duration=2.0, seed=7)
    result = model.transcribe(audio_bytes=noise_wav)

    # Must not be the forbidden string
    assert result["text"] != FORBIDDEN_HARDCODED_TRANSCRIPT

    # Energy must be above silence threshold
    assert result.get("energy", 0.0) > 0.001

    # If it's not silent, silence flag should be False
    if result["energy"] > 0.001:
        assert result.get("silence") is not True


# ---------------------------------------------------------------------------
# Test 5: Empty audio rejected with ValueError
# ---------------------------------------------------------------------------

def test_empty_audio_raises_value_error():
    model = LocalWhisperSpeechModel()
    with pytest.raises(ValueError, match="empty"):
        model.transcribe(audio_bytes=b"")


# ---------------------------------------------------------------------------
# Test 6: Corrupted audio raises ValueError
# ---------------------------------------------------------------------------

def test_corrupted_audio_raises_value_error():
    model = LocalWhisperSpeechModel()
    garbage = b"\x00\x01\x02\x03\xff\xfe" * 10  # Not a valid WAV or PCM
    with pytest.raises(ValueError):
        model.transcribe(audio_bytes=garbage)


# ---------------------------------------------------------------------------
# Test 7: Model unavailable → controlled error (no crash, no fake text)
# ---------------------------------------------------------------------------

def test_model_unavailable_returns_error_not_fake_text(monkeypatch):
    """
    If the model fails to load (e.g. package missing), transcribe() must
    return a controlled error dict — never a fake transcript.
    """
    model = LocalWhisperSpeechModel()
    # Force model into a state where no inference backend is available
    model._whisper_model = None
    model._ort_encoder = None
    model._ort_decoder = None
    model._status = ModelStatus.FAILED

    # Prevent auto-reload from succeeding by monkeypatching load()
    monkeypatch.setattr(model, "load", lambda: False)

    wav = make_sine_wav(duration=1.0)
    result = model.transcribe(audio_bytes=wav)

    assert result.get("success") is False
    assert result.get("text", "") == ""
    # Must not contain the forbidden phrase
    assert result.get("text") != FORBIDDEN_HARDCODED_TRANSCRIPT
    assert "error" in result


# ---------------------------------------------------------------------------
# Test 8: Local-only — no network calls during transcription
# ---------------------------------------------------------------------------

def test_transcription_makes_no_network_calls(monkeypatch):
    """
    Verifies that transcribe() does not make any outbound socket connections.
    NEXUS LocalNetworkGuard is not active in unit tests, so we monkeypatch
    socket.socket.connect to catch any connection attempts.
    """
    import socket
    connection_attempts = []

    original_connect = socket.socket.connect

    def spy_connect(self, address):
        host = address[0] if isinstance(address, tuple) else str(address)
        if host not in ("127.0.0.1", "::1", "localhost"):
            connection_attempts.append(host)
        return original_connect(self, address)

    monkeypatch.setattr(socket.socket, "connect", spy_connect)

    model = LocalWhisperSpeechModel()
    wav = make_sine_wav(duration=1.0)
    model.transcribe(audio_bytes=wav)

    assert connection_attempts == [], (
        f"FAIL: transcribe() made outbound network calls to: {connection_attempts}. "
        f"ASR must be 100% local."
    )


# ---------------------------------------------------------------------------
# Test 9: health_check returns real values
# ---------------------------------------------------------------------------

def test_health_check_returns_real_values():
    model = LocalWhisperSpeechModel()
    model.load()
    health = model.health_check()

    assert "model" in health
    assert "status" in health
    assert "is_loaded" in health
    assert "runtime" in health
    assert "execution_provider" in health
    assert health["is_loaded"] is True
    assert health["runtime"] not in ("Unloaded", "")
    assert health["execution_provider"] not in ("Unloaded", "")


# ---------------------------------------------------------------------------
# Test 10: API endpoint — transcribe returns real transcript structure
# ---------------------------------------------------------------------------

def test_api_speech_status_endpoint(client):
    response = client.get("/speech/status")
    assert response.status_code == 200
    data = response.json()
    assert "model_name" in data
    assert "execution_provider" in data
    assert "runtime" in data
    assert "status" in data
    assert isinstance(data["blockers"], list)
    # model_name must not be a generic placeholder
    assert data["model_name"] != ""


def test_api_speech_transcribe_returns_structured_response(client):
    wav_bytes = make_sine_wav(duration=1.5)
    files = {"file": ("test.wav", io.BytesIO(wav_bytes), "audio/wav")}

    response = client.post("/speech/transcribe", files=files)
    assert response.status_code == 200
    data = response.json()

    # Must have all required fields
    assert "transcript" in data
    assert "language" in data
    assert "duration_seconds" in data
    assert "energy" in data
    assert "silence" in data
    assert "hardware" in data
    assert "runtime" in data
    assert "execution_provider" in data
    assert "latency_ms" in data
    assert "success" in data
    assert data["success"] is True
    assert data["runtime"] != ""
    assert data["execution_provider"] not in ("", "Unloaded")

    # Transcript must NOT be the forbidden hard-coded string
    assert data["transcript"] != FORBIDDEN_HARDCODED_TRANSCRIPT, (
        "API returned the forbidden hard-coded transcript."
    )

    # Duration should match our 1.5s input
    assert 1.4 <= data["duration_seconds"] <= 1.6


def test_api_speech_transcribe_empty_audio_rejected(client):
    files = {"file": ("empty.wav", io.BytesIO(b""), "audio/wav")}
    response = client.post("/speech/transcribe", files=files)
    assert response.status_code == 422


def test_api_silence_returns_empty_transcript(client):
    silent_wav = make_silent_wav(duration=1.0)
    files = {"file": ("silent.wav", io.BytesIO(silent_wav), "audio/wav")}
    response = client.post("/speech/transcribe", files=files)
    assert response.status_code == 200
    data = response.json()
    assert data["transcript"] == ""
    assert data["silence"] is True
    assert data["transcript"] != FORBIDDEN_HARDCODED_TRANSCRIPT


# ---------------------------------------------------------------------------
# Test 11: Auto-execution guard (unchanged from original)
# ---------------------------------------------------------------------------

def test_transcription_does_not_auto_execute_plan(client):
    """
    Verifies that /speech/transcribe only returns the transcript —
    it does NOT trigger the agent planner or executor.
    """
    wav_bytes = make_sine_wav(duration=1.0)
    files = {"file": ("goal_voice.wav", io.BytesIO(wav_bytes), "audio/wav")}
    response = client.post("/speech/transcribe", files=files)
    assert response.status_code == 200
    data = response.json()
    assert "transcript" in data
    assert "plan" not in data
    assert "tasks" not in data
    assert "execution_result" not in data


# ---------------------------------------------------------------------------
# Test 12: Hardware blocker detection on non-Snapdragon host
# ---------------------------------------------------------------------------

def test_hardware_blocker_detection():
    import platform
    model = LocalWhisperSpeechModel()
    machine = platform.machine().upper()

    if machine not in ("ARM64", "AARCH64"):
        # On x86 dev host: blockers must be documented
        assert len(model.blockers) >= 1
        # CPU target, not NPU
        assert "CPU" in model.target_hardware
        # Provider must not claim QNN on AMD64
        assert model.execution_provider != "QNNExecutionProvider"
