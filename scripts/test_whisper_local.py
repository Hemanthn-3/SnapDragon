#!/usr/bin/env python3
"""
NEXUS Local ASR Manual Test Script (Phase 17)
Tests real Whisper inference on actual audio files or generates synthetic audio.

Usage:
    python scripts/test_whisper_local.py                   # Uses synthetic audio
    python scripts/test_whisper_local.py path/to/audio.wav # Uses your audio file

Output:
    NEXUS LOCAL ASR TEST
    Model: ...
    Runtime: ...
    Execution provider: ...
    Latency: ... ms
    RTF: ...
    Transcript: ...
"""

import io
import sys
import time
import wave
from pathlib import Path

import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def make_test_wav(duration: float = 3.0, frequency: float = 440.0, sample_rate: int = 16000) -> bytes:
    """Generates a pure sine tone WAV for baseline testing."""
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    samples = (np.sin(2 * np.pi * frequency * t) * 0.5 * 32767).astype(np.int16)
    bio = io.BytesIO()
    with wave.open(bio, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(samples.tobytes())
    return bio.getvalue()


def load_wav_file(path: str) -> bytes:
    with open(path, "rb") as f:
        return f.read()


def print_header():
    print()
    print("=" * 60)
    print("   NEXUS LOCAL ASR TEST")
    print("=" * 60)


def print_section(title: str, value: str):
    print(f"\n{title}")
    print(f"  {value}")


def run_test(audio_bytes: bytes, audio_label: str):
    from backend.models_local.whisper_speech import LocalWhisperSpeechModel

    model = LocalWhisperSpeechModel()

    print(f"\n[Loading model...]")
    t_load = time.perf_counter()
    ok = model.load()
    load_ms = round((time.perf_counter() - t_load) * 1000, 1)

    if not ok:
        print("ERROR: Model failed to load.")
        print(f"  Health: {model.health_check()}")
        return

    print(f"  Load time: {load_ms} ms")

    print(f"\n[Running inference on: {audio_label}]")
    result = model.transcribe(audio_bytes=audio_bytes)

    print_header()

    print_section("Model:", result.get("model", "unknown"))
    print_section("Runtime:", result.get("runtime", "unknown"))
    print_section("Execution provider:", result.get("execution_provider", "unknown"))

    latency_ms = result.get("latency_ms", 0.0)
    duration_s = result.get("duration_seconds", 0.0)
    rtf = (latency_ms / 1000.0) / duration_s if duration_s > 0 else float("inf")

    print_section("Audio duration:", f"{duration_s:.2f} sec")
    print_section("Latency:", f"{latency_ms:.1f} ms")
    print_section("Real-time factor (RTF):", f"{rtf:.3f}  (< 1.0 = faster than real-time)")
    print_section("Energy (RMS):", f"{result.get('energy', 0.0):.5f}")
    print_section("Silence detected:", str(result.get("silence", False)))

    if result.get("success") is False:
        print_section("ERROR:", result.get("error", "Unknown error"))
    else:
        transcript = result.get("text", "")
        print_section("Transcript:", f'"{transcript}"' if transcript else "[empty — silence or no speech]")
        print_section("Language:", result.get("language", "unknown"))

    if result.get("blockers"):
        print("\nHardware Blockers (NPU not available):")
        for b in result["blockers"]:
            print(f"  - {b}")

    print()
    print("=" * 60)

    # Verify the forbidden hard-coded string is not present
    forbidden = "Analyze these inspection documents and create an action report."
    if result.get("text") == forbidden:
        print("CRITICAL FAIL: Output is the hard-coded fake transcript!")
        print("The real ASR implementation is not active.")
        sys.exit(1)
    else:
        print("CONTRACT CHECK: Hard-coded transcript not present. [PASS]")

    print("=" * 60)


def main():
    if len(sys.argv) > 1:
        audio_path = sys.argv[1]
        print(f"\nLoading audio from: {audio_path}")
        try:
            audio_bytes = load_wav_file(audio_path)
            audio_label = Path(audio_path).name
        except FileNotFoundError:
            print(f"ERROR: File not found: {audio_path}")
            sys.exit(1)
    else:
        print("\nNo audio file provided. Using synthetic 440 Hz tone (3 seconds).")
        print("TIP: Pass a real speech WAV file as an argument for meaningful output:")
        print("     python scripts/test_whisper_local.py my_recording.wav")
        audio_bytes = make_test_wav(duration=3.0, frequency=440.0)
        audio_label = "synthetic_440hz_3s.wav"

    run_test(audio_bytes, audio_label)


if __name__ == "__main__":
    main()
