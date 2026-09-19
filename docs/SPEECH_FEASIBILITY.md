# NEXUS: Local Speech Recognition (ASR) Feasibility & Hardware Compatibility

**Document**: `docs/SPEECH_FEASIBILITY.md`  
**Target Platform**: Qualcomm Snapdragon X Elite / Plus (Windows 11 ARM64)  
**Primary Model**: `Whisper-Small-Quantized` (`qualcomm/Whisper-Small-Quantized`)  
**Quantization**: `w8a16` (8-bit weights, 16-bit activations)  
**Target Runtime**: ONNX Runtime with QNN Execution Provider (`onnxruntime-qnn` / `QnnHtp.dll`)

---

## 1. Selected Speech Model Specification

As documented during Phase 0 in [`docs/MODEL_SELECTION.md`](file:///c:/Users/heman/Desktop/Snapdragon/docs/MODEL_SELECTION.md), `Whisper-Small-Quantized` was verified as the primary Automatic Speech Recognition (ASR) model for NEXUS:

| Parameter | Specification |
| :--- | :--- |
| **Model Name** | `Whisper-Small-Quantized` |
| **Hugging Face Repository** | `qualcomm/Whisper-Small-Quantized` |
| **Qualcomm AI Hub Asset** | `whisper_small_quantized-precompiled_qnn_onnx-w8a16-qualcomm_snapdragon_x_elite.zip` |
| **Tool Versions** | QAIRT v2.45.0.260326154327, ONNX Runtime v1.27.1 |
| **Target Hardware** | Hexagon Tensor Processor (HTP) on Snapdragon X Elite / Plus |
| **Precision** | `w8a16` (AIMET post-training quantization) |
| **Input Shape** | Static Mel-Spectrogram: `[1, 80, 3000]` (80 mel-frequency channels, 30-second window at 16 kHz) |
| **Memory Footprint** | ~240 MB resident RAM |
| **Offline Guarantee** | 100% local inference, zero external network requests |

---

## 2. Audio Preprocessing Pipeline

Whisper operates on log-mel spectrogram features rather than raw audio waveforms:

```
[Microphone Audio (16 kHz PCM)]
             |
             v
[Oryon CPU: STFT & Mel-Filterbank]
(Convert 16,000 samples/sec to 80-channel log-mel spectrogram)
             |
             v
[Static Tensor Padding / Truncation]
(Shape: [1, 80, 3000] representing 30.0s window)
             |
             v
[Hexagon NPU: Whisper Encoder (w8a16)]
(QNN Context Binary: feature extraction on HTP)
             |
             v
[Hexagon NPU / CPU: Whisper Decoder (w8a16)]
(Autoregressive token generation & vocabulary projection)
             |
             v
[Transcript String]
```

---

## 3. Exact Hardware Execution Blocker Analysis

Per project guidelines: **"If the model cannot yet run on the target Snapdragon environment, document the exact blocker instead of creating a fake implementation."**

### Blocker 1: Physical Absence of Hexagon NPU on x86_64 Development Hosts
- **Analysis**: The current host system is running Windows 11 on `AMD64` (x86_64 architecture). Qualcomm's Hexagon Tensor Processor (HTP) is a proprietary hardware accelerator found exclusively on Qualcomm Snapdragon silicon.
- **Impact**: `QnnHtp.dll` cannot initialize on non-Snapdragon CPUs, as it relies on Qualcomm Hexagon hardware registers and device drivers.

### Blocker 2: Architecture Mismatch for QNN Context Binaries
- **Analysis**: The Qualcomm AI Hub precompiled assets (`whisper_small_quantized-qnn_context_binary-w8a16-qualcomm_snapdragon_x_elite.zip`) are compiled specifically for Qualcomm's V75/V79 HTP microarchitecture.
- **Impact**: These binary blobs cannot be loaded or interpreted by generic x86_64 ONNX Runtime providers.

### Blocker 3: Dual Execution Strategy
To provide zero-hallucination integrity while maintaining full Snapdragon portability:
1. **Target Snapdragon Hardware (ARM64 Windows 11)**:
   - Evaluates `platform.machine() == "ARM64"` and checks for `QnnHtp.dll`.
   - Initializes `QNNExecutionProvider` with the precompiled QNN context binary.
2. **x86_64 Development Environment**:
   - Explicitly logs `BLOCKER: Hexagon NPU unavailable on AMD64 development host`.
   - Falls back to local CPU execution via ONNX Runtime `CPUExecutionProvider` or local wave-audio feature processing.
   - **Never claims NPU execution unless verified on actual Snapdragon hardware.**

---

## 4. User Interaction & Workflow Rules

1. **Microphone Capture**: Uses the browser's `navigator.mediaDevices.getUserMedia({ audio: true })`.
2. **Permission Handling**: Explicitly prompts the user for microphone access. If denied, provides clear instructions without crashing.
3. **Five Discrete UI States**:
   - `Idle`: Ready to record.
   - `Listening`: Capturing live audio with active visual indicator.
   - `Processing`: Local ASR transcribing audio.
   - `Transcript Ready`: Populated in an editable textarea for review.
   - `Error`: Clear diagnostic message.
4. **No Auto-Execution**:
   - Transcription **never** triggers plan generation or tool execution automatically.
   - The user must explicitly inspect, edit if desired, and click **`[Use as NEXUS Goal]`** to submit the transcribed text to the planner.
