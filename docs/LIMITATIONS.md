# NEXUS Known Limitations & Future Work

**Version**: 1.0.0  
**Scope**: Honest engineering assessment for competition submission

---

> [!NOTE]
> This document exists to provide an accurate, honest view of NEXUS limitations. Transparency about what the system cannot do is as important as demonstrating what it can.

---

## 1. Current Development Environment

### 1.1 Physical Snapdragon Hardware Not Used for Benchmarking

**Status**: PARTIAL  
**Detail**: The benchmark suite (`benchmarks/suite.py`) was executed on the development host (Intel Core i7, AMD64, Windows 11 Pro). Measured latencies reflect CPU-only ONNX Runtime execution.

**Implication**: 
- Reported LLM latency (696ms warm median for 16 tokens) is a CPU baseline, not NPU performance.
- On Snapdragon X Elite with `QNNExecutionProvider`, LLM throughput is expected to improve significantly (target: 40–48 tokens/second on Hexagon NPU per Qualcomm specifications).
- Reported embedding latency (48ms), vision latency (44ms) are also CPU baselines.

**What Is Real**: All model architectures, quantization formats, ONNX graph structures, and provider selection logic are production-ready for Snapdragon NPU execution. The `detect_execution_provider()` adapter will automatically activate `QNNExecutionProvider` on Snapdragon hardware.

---

## 2. OCR Implementation

### 2.1 Full EasyOCR NPU Pipeline Pending

**Status**: NOT_YET_IMPLEMENTED (clearly marked in UI and benchmarks)  
**Detail**: `RunOCRTool` currently reports `NOT_YET_IMPLEMENTED` rather than producing fabricated OCR results. EasyOCR's CRAFT text detection model and CRNN recognition backbone are architecturally compatible with Qualcomm AI Hub QNN export, but the compiled DLC context binary for this host has not been generated.

**Implication**: The demo workflow uses deterministic simulated OCR output for the competition demo (`"Fastener Bolt Torque Applied: 95 Nm"`). Real OCR on live documents requires the compiled CRAFT+CRNN QNN context binary.

**Path to Resolution**: Run `python -m qai_hub_models.models.easyocr.export --target-runtime onnx --device "Snapdragon X Elite CRD"` on a Qualcomm AI Hub account to generate the compiled binary.

---

## 3. Language Model Capability Boundaries

### 3.1 1B Parameter Context Limitations

**Status**: KNOWN  
**Detail**: `Llama-3.2-1B-Instruct` (w4a16) is a 1.2 billion parameter model. While fast, it has:
- Maximum context window: 2048 tokens (fixed at NPU compilation time)
- JSON formatting reliability: High with few-shot prompting, but occasional malformed outputs on complex nested structures
- Complex multi-document reasoning: Adequate for 3–5 document packages; accuracy degrades on very large document corpora

**Mitigation**: The planner includes JSON extraction fallback (`extract_json_payload()`) and malformed output detection. Plans that fail validation are rejected cleanly with user-facing error messages.

### 3.2 Hallucination Risk on Unsupported Claims

**Status**: MITIGATED (not eliminated)  
**Detail**: The LLM reasoning layer (`create_report` tool) can generate plausible-sounding findings not strictly supported by retrieved evidence chunks. The verification subsystem detects and flags these, but cannot guarantee detection of all hallucinations.

**Mitigation**: All findings are tagged `OBSERVED` (from retrieved text) or `INFERRED` (from model reasoning). Verification marks claims as `SUPPORTED`, `CONTRADICTED`, or `UNVERIFIED`. Users should treat `INFERRED` + `UNVERIFIED` items as requiring human review before action.

---

## 4. Voice / ASR

### 4.1 Whisper Small Accuracy on Technical Jargon

**Status**: KNOWN  
**Detail**: Whisper-Small (w8a16) provides good general ASR but has reduced accuracy on:
- Dense engineering terminology (torque specifications, chemical compound names)
- Strong accents or background noise
- Very quiet microphone input

**Mitigation**: All voice input is shown in the text field before submission, allowing manual correction. The text input fallback is always available.

### 4.2 ASR on Long Commands

**Status**: KNOWN  
**Detail**: Whisper-Small processes 30-second audio windows. Commands longer than ~30 seconds may be truncated. In practice, goal statements are typically under 10 seconds.

---

## 5. Security Boundaries

### 5.1 Python-Level Socket Guard Only

**Status**: KNOWN  
**Detail**: `LocalNetworkGuard` intercepts Python `socket` module calls. Native C extensions that bypass Python's socket layer via direct Win32 API calls (`ws2_32.dll`) would not be intercepted.

**Mitigation**: NEXUS uses only verified open-source Python packages. For high-security deployments, supplement with Windows Firewall egress rules.

### 5.2 No OS-Level Process Sandbox

**Status**: KNOWN  
**Detail**: NEXUS runs as a standard Windows user process. A memory corruption exploit in an underlying C library (onnxruntime, pypdf, Pillow) would operate with current user privileges.

**Future Work**: Windows AppContainer isolation or Windows Sandbox deployment.

### 5.3 Sophisticated Multi-Document Injection

**Status**: PARTIALLY MITIGATED  
**Detail**: Simple prompt injections ("Ignore all previous instructions") are reliably blocked by fencing and Rule 8. Highly nuanced, split-across-chunks injections that subtly bias reasoning without using obvious trigger phrases cannot be guaranteed blocked by a 1B parameter model.

**Mitigation**: NEXUS is "propose-only" — all destructive actions (file export) require explicit human approval.

---

## 6. Performance

### 6.1 High CPU Utilization on Development Host

**Status**: EXPECTED (development baseline)  
**Detail**: On the AMD64 development host, embedding, vision, and LLM operations consume 650–1400% CPU (multi-core utilization). This is CPU-only inference without NPU offload.

**On Snapdragon Target**: Matrix operations shift to the 45 TOPS Hexagon NPU, freeing the Oryon CPU cores for desktop work.

### 6.2 Memory Footprint

**Status**: WITHIN SPEC  
**Detail**: Full workflow RSS peaks at ~2.55 GB on the development host. This includes ONNX Runtime sessions for all 5 models simultaneously resident in memory.

**On Snapdragon**: With QNN context binaries, model memory footprint is approximately 1.89 GB (as documented in `docs/MODEL_SELECTION.md`).

---

## 7. Demo-Specific Limitations

### 7.1 Synthetic Data Only

**Status**: BY DESIGN  
**Detail**: The competition demo uses 100% synthetic industrial inspection data from `demo_data/`. It is not validated against a real industrial equipment database or real inspection standards.

**Purpose**: Reproducibility, determinism, and demonstration of NEXUS cognitive capabilities without requiring proprietary data.

### 7.2 Report Quality on Real Documents

**Status**: USE CASE DEPENDENT  
**Detail**: Report quality on real-world documents depends on document clarity, OCR accuracy, embedding relevance, and LLM capability. The competition demo represents an idealized scenario with high-contrast synthetic documents.

---

## 8. Future Work Roadmap

| Priority | Item | Effort |
|---|---|---|
| High | Compile EasyOCR DLC for Snapdragon X Elite HTP | Medium |
| High | Run full benchmark suite on physical Snapdragon X Elite hardware | Low (hardware only) |
| High | Windows AppContainer sandbox for backend process | High |
| Medium | Upgrade to Llama-3.2-3B for higher reasoning quality | Low |
| Medium | Streaming token output in UI (real-time LLM output) | Medium |
| Medium | Long-document chunking pipeline (>50 pages) | Medium |
| Low | Multi-session conversation memory | High |
| Low | Screen capture & passive context ingestion | High |
| Low | Meeting transcription (live audio stream) | Medium |
