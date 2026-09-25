# NEXUS Whisper Replacement Plan — Phase 17

**Date**: September 2026  
**Status**: APPROVED FOR IMPLEMENTATION

---

## 1. Current Implementation

File: `backend/models_local/whisper_speech.py`

### What the current code does

The `LocalWhisperSpeechModel` class:

1. `__init__()` — Detects ARM64 vs x86_64 and records hardware blockers. Does **not** load any model.
2. `load()` — Queries `onnxruntime.get_available_providers()` and sets `_execution_provider`. Does **not** load any Whisper model weights. Returns `True` unconditionally unless an exception is thrown importing `onnxruntime`.
3. `_parse_audio()` — **Correctly** parses WAV or raw PCM bytes into a normalized float32 numpy array. This code is sound and will be preserved.
4. `transcribe()` — **BROKEN**: Calls `_parse_audio()` to compute RMS energy, then:
   - If `energy < 0.001` → returns `"[Silence / Inaudible Audio]"` (correct behavior)
   - Otherwise → **hard-codes** `"Analyze these inspection documents and create an action report."` regardless of what was actually said.

### Why this is not real ASR

- No Whisper model file is loaded at any point.
- The transcript does not change with the spoken content.
- Any non-silent audio — a dog bark, keyboard noise, music, or any spoken sentence — returns the identical predetermined string.
- The `load()` method succeeds even with no model files present on disk.

---

## 2. Why It Must Be Replaced

Phase 17 requirement: **"Different audio must produce different transcripts."**

The current system violates this. Audio A ("Hello, this is a test") and Audio B ("Analyze the report") both return the same fixed string. This is fundamentally broken behavior that misleads users and falsifies system capabilities.

---

## 3. Selected Replacement Model

### Model: `openai/whisper-small`

**Rationale**:

| Criterion | Status |
|---|---|
| Real speech recognition | ✅ YES — actual neural ASR |
| Local execution | ✅ YES — runs entirely on-device |
| No cloud API calls | ✅ YES — all inference is local |
| Available on dev host | ✅ YES — PyTorch 2.12.0+cpu is installed |
| Qualcomm AI Hub variant | `Whisper-Small-Quantized` (w8a16) — deployment target |
| Cross-platform | ✅ YES — ARM64 and x86_64 |

**Runtime**: `openai-whisper` Python package (uses PyTorch for inference)

**Package**: `openai-whisper` (pip installable, MIT license)

**Model size**: ~242 MB (downloaded once to disk cache on first use)

---

## 4. Runtime and Execution Path

### CPU Path (Development Host — AMD64)

```
audio bytes
    │
    ▼
_parse_audio()  [existing, preserved]
    │  WAV/PCM → float32 numpy array
    │  stereo → mono downmix
    │  any sample rate → 16 kHz resampling via scipy/numpy
    │
    ▼
whisper.load_model("small")  [openai-whisper]
    │  Loads weights from disk cache (~242 MB)
    │  Runs on PyTorch CPUExecutionProvider
    │
    ▼
model.transcribe(audio_array, fp16=False)
    │  Real Whisper inference
    │  Returns segments + detected language
    │
    ▼
TranscriptResult {text, language, segments, latency_ms}
```

### Snapdragon / QNN Path (ARM64 Target)

```
audio bytes
    │
    ▼
_parse_audio()  [same preprocessing]
    │
    ▼
onnxruntime.InferenceSession(
    "models/speech/Whisper-Small-Quantized/encoder.onnx",
    providers=["QNNExecutionProvider", "CPUExecutionProvider"]
)
    │  QNN context binary loaded on Hexagon HTP
    │  Encoder: mel-spectrogram → hidden states (NPU)
    │  Decoder: autoregressive token generation (NPU)
    │
    ▼
Greedy token decode → transcript
```

**QNN path status**: Architected and coded, but NOT YET VERIFIED on physical Snapdragon hardware. The code detects `QNNExecutionProvider` at runtime and uses it when present. On AMD64, it falls back to `openai-whisper` (CPU).

---

## 5. Audio Preprocessing Pipeline

```
Input: bytes (WAV RIFF, or raw PCM)
    │
    ├─ _parse_audio() [preserved from existing code]
    │      WAV header parse → int16/float32 → normalize to [-1.0, +1.0]
    │      Multi-channel → mono (mean across channels)
    │      Returns: (float32 array, sample_rate, duration_seconds)
    │
    ▼
Silence detection: RMS energy < 0.001
    │
    ├─ YES → return {"text": "", "silence": true}  [NOT a fake transcript]
    │
    └─ NO → continue
    │
    ▼
Resampling to 16 kHz (if needed)
    │  scipy.signal.resample_poly(samples, 16000, original_sr)
    │
    ▼
Whisper mel-spectrogram preprocessing
    │  whisper.audio.log_mel_spectrogram(samples)
    │  [handled internally by openai-whisper]
    │
    ▼
Inference → transcript text
```

**Required sample rate**: 16 kHz (Whisper requirement — documented)  
**Bit depth**: 16-bit PCM → normalized float32 (handled)  
**Max duration**: 30 seconds per chunk (Whisper's fixed input window)

---

## 6. Input and Output Format

### Input
```python
transcribe(audio_bytes: bytes, sample_rate: int = 16000) -> Dict[str, Any]
```

- `audio_bytes`: WAV (RIFF) or raw 16-bit PCM bytes
- `sample_rate`: hint for raw PCM interpretation (WAV header takes precedence)

### Output
```json
{
  "text": "actual transcribed speech here",
  "language": "en",
  "duration_seconds": 3.2,
  "sample_rate": 16000,
  "energy": 0.0412,
  "model": "openai/whisper-small",
  "runtime": "openai-whisper",
  "execution_provider": "CPUExecutionProvider",
  "latency_ms": 1240,
  "success": true
}
```

### Error Output
```json
{
  "success": false,
  "error": "Speech recognition inference failed: [reason]",
  "text": ""
}
```

No hard-coded transcripts are returned. Errors are explicit.

---

## 7. Snapdragon NPU Path (Documented)

On Snapdragon X Elite with `QNNExecutionProvider` available:

- ONNX model files loaded from `models/speech/Whisper-Small-Quantized/`
- Encoder and decoder compiled to QNN context binaries (`.bin`) via Qualcomm AI Hub
- Provides ~4× latency improvement over CPU inference
- Reports `"execution_provider": "QNNExecutionProvider"` in output

**VERIFIED ON DEVELOPMENT MACHINE**: NO  
**QUALCOMM-DOCUMENTED**: YES (Qualcomm AI Hub `Whisper-Small-Quantized` model card)  
**SUPPORTED BY CODE**: YES (runtime detection + provider selection)  
**NOT YET VERIFIED ON SNAPDRAGON HARDWARE**: confirmed

---

## 8. Remaining Limitations

1. **First-run model download**: `openai-whisper` downloads model weights (~242 MB) on first call. Subsequent runs use disk cache. For fully offline deployment, the model must be pre-downloaded.
2. **Inference speed on CPU**: Whisper-Small on CPU takes approximately 0.5–3× real-time depending on host. For a 3-second voice command, expect 1–3 second processing time.
3. **30-second max input**: Whisper processes audio in 30-second windows. Audio longer than 30 seconds is truncated. For NEXUS voice commands this is not a practical limitation.
4. **Language detection**: Whisper auto-detects language. For English-only deployments, passing `language="en"` speeds up inference slightly.
5. **MediaRecorder audio format**: Browsers output `audio/webm;codecs=opus` from `MediaRecorder`, not raw WAV. The backend now handles both via `pydub`/`soundfile` fallback, or the audio is decoded via scipy.

---

## 9. Files Modified

| File | Change |
|---|---|
| `backend/models_local/whisper_speech.py` | **Full replacement** — real openai-whisper inference |
| `backend/interfaces/speech.py` | Add `latency_ms`, `runtime`, `success` to return contract |
| `backend/routes_speech.py` | Add `latency_ms`, `runtime` to response schema |
| `tests/test_speech.py` | Replace fake-asserting tests with real behavioral tests |
| `scripts/test_whisper_local.py` | **New** — manual real-world inference test |
| `benchmarks/suite.py` | Add real ASR benchmark workload |
| `requirements.txt` | Add `openai-whisper` |

---

## 10. Definition of Done

- [x] `openai-whisper` package installed and imported
- [ ] `load()` actually loads Whisper model weights
- [ ] `transcribe()` calls real Whisper inference
- [ ] Hard-coded string on line 186 removed
- [ ] Different audio → different transcript (verified by test)
- [ ] Silence → empty string (not fake transcript)
- [ ] Invalid audio → explicit error dict
- [ ] Model unavailable → explicit error (no crash)
- [ ] Local-only verified (no network calls during inference)
- [ ] `latency_ms` measured and returned
- [ ] `execution_provider` accurately reported
- [ ] ASR benchmark added
- [ ] All 137 existing tests still pass
