# NEXUS: Minimum Viable Architecture Proposal

**Project**: NEXUS — Offline-First Multimodal AI Work Agent  
**Design Philosophy**: *"Smallest technically reliable system that can demonstrate a real Snapdragon advantage."*  
**Target Platform**: Snapdragon X Elite, Snapdragon X Plus, Snapdragon X2 Series  
**Target Operating System**: Windows 11 on ARM64 (24H2+ / Copilot+ PC)

---

## 1. Executive Concept & System Mission

NEXUS is an autonomous, air-gapped, offline-first work assistant tailored specifically for Snapdragon-powered Copilot+ PCs. It continuously observes desktop context (screen state, active documents, meeting audio) and translates high-level multimodal voice and text requests into deterministic desktop actions and synthesized insights.

Rather than overloading the system with unverified experimental frameworks or heavy 7B–14B models that cause memory thrashing, NEXUS is architected around a **tight, decoupled, 5-stage pipeline** where **every single machine learning operation is offloaded to the 45 TOPS Qualcomm Hexagon NPU**.

```
+=======================================================================================+
|                                    NEXUS WORK AGENT                                   |
+=======================================================================================+
|                                                                                       |
|  [USER INPUT]                [DESKTOP CONTEXT]               [AUDIO INGESTION]        |
|  Keyboard / Mic Trigger      Active Window / Screen Capture  Microphone / Meeting     |
|         |                            |                                |               |
+---------|----------------------------|--------------------------------|---------------+
          |                            |                                |
          |                            v                                v
+---------|--------------------+--------------------------------+---------------+
|         v                    |      HEXAGON NPU PIPELINE      |               |
|  Voice Command               |                                |               |
|  [Whisper-Small w8a16]       |  [EasyOCR w8a16]               |               |
|  NPU ASR Latency: < 120ms    |  NPU OCR Latency: < 140ms      |               |
|         |                    |                                |               |
|         +------------------->|  [OpenAI-CLIP w8a16]           |               |
|                              |  NPU Vision Latency: < 35ms    |               |
|                              |                                |               |
|                              |  [all-MiniLM-L6-v2 w8a16]      |               |
|                              |  NPU Embed Latency: < 15ms     |               |
+------------------------------+--------------------------------+---------------+
                                       |
                                       v
          +-------------------------------------------------------------+
          |             LOCAL SEMANTIC WORKSPACE MEMORY                 |
          |       In-Memory Vector Store + Active Window State          |
          +-------------------------------------------------------------+
                                       |
                                       v
+-------------------------------------------------------------------------------+
|                       NPU REASONING & DISPATCH ENGINE                         |
|                                                                               |
|             [Llama-v3.2-1B-Instruct (w4a16) via ORT GenAI]                    |
|             - Token Rate: 42-48 tokens/sec on Hexagon NPU                     |
|             - Output: Strictly Formatted Structured JSON Action Plan          |
+-------------------------------------------------------------------------------+
                                       |
                                       v
+-------------------------------------------------------------------------------+
|                      WINDOWS DETERMINISTIC ACTION ENGINE                      |
|                                                                               |
|  [File Management]    [Clipboard Synthesis]    [App Automation]    [Notifs]   |
|  Read/Write/Summarize Format & Inject Notes    Focus Window/Paste  OS Toast   |
+===============================================================================+
```

---

## 2. Core Architectural Components

### 2.1 Perceptual Ingestion Subsystem (Sensory Layer)
The sensory layer gathers real-time desktop signals without interrupting the user.
1. **Audio Stream Capture**:
   - Captures microphone input (voice commands) or system audio loopback (virtual meetings) using native Windows CoreAudio (WASAPI).
   - Audio is buffered into rolling 30-second frames and converted into mel-spectrogram arrays using optimized ARM64 NEON routines on Oryon CPU.
2. **Screen & Window State Capture**:
   - Uses the Windows Graphics Capture API (`Windows.Graphics.Capture`) to capture full-screen or focused application window frames with zero GPU blit overhead.
   - Downsamples dynamically: high-resolution crops for text areas; 224x224 RGB tensors for visual classification.

### 2.2 NPU Perceptual Preprocessing (The 45 TOPS Acceleration Tier)
All heavy perceptual operations are dispatched directly to the Qualcomm Hexagon NPU via **ONNX Runtime QNN Execution Provider** (`QnnHtp.dll`):
1. **Speech-to-Text (`Whisper-Small-Quantized`, `w8a16`)**:
   - Converts audio frames into text transcripts in sub-120ms per 5-second chunk.
   - Allows users to speak naturally to command the desktop ("Summarize what is currently on my screen and write a reply to Alice").
2. **Text & Document Extraction (`EasyOCR`, `w8a16`)**:
   - Detects and extracts text from documents, browser tabs, chat windows, and IDEs.
   - Produces localized text tokens paired with pixel bounding coordinates `(x, y, w, h)`.
3. **Visual Semantic Representation (`OpenAI-Clip`, `w8a16`)**:
   - Encodes the visual desktop frame into a 512-dimensional vector.
   - Categorizes whether the user is viewing code, a slide presentation, a spreadsheet, a browser research paper, or a video call.
4. **Vector Embeddings (`all-MiniLM-L6-v2`, `w8a16`)**:
   - Converts extracted screen text, voice commands, and local workspace documents into 384-dimensional embeddings.
   - Latency on Hexagon NPU is **< 15ms per chunk**, allowing real-time semantic indexing without slowing down the desktop.

### 2.3 Local Semantic Workspace Memory
- An in-memory, lightweight vector index (flat inner-product / cosine similarity) running in the NEXUS local process.
- Retains:
  - Rolling screen state history (last 10 minutes of active work context).
  - Indexed workspace files (e.g., project documentation, notes, recent meeting transcripts).
- Zero cloud database dependencies: completely resident in unified LPDDR5x RAM.

### 2.4 Reasoning & Planning Engine (Local SLM)
- **Model**: `Llama-v3.2-1B-Instruct` (Quantized `w4a16`).
- **Runtime**: `onnxruntime-genai` with `QNNExecutionProvider`.
- **Function**:
  - Ingests:
    1. User's intent (voice command or typed query).
    2. Retrieved screen text and window title from the semantic memory.
    3. Available action schema.
  - Generates:
    - Deterministic JSON action payloads containing tool calls (e.g., `write_file`, `copy_clipboard`, `open_url`, `summarize_meeting`, `notify_user`).
- **NPU Generation Speed**: **42–48 tokens/second** on Snapdragon X Elite/Plus Hexagon NPU.
- **Latency to First Token**: **< 120ms**.

### 2.5 Windows Deterministic Action Engine
The execution tier does not rely on nondeterministic keyboard macros; it utilizes standard Windows 11 APIs:
1. **File Operations**: Atomic file reading, diff generation, and workspace saving.
2. **Clipboard Manager**: Formats synthesized notes, action items, or code snippets and pushes them directly to the Windows Clipboard.
3. **UI Automation**: Uses UI Automation (`UIAutomationClient`) and Win32 APIs (`SetForegroundWindow`, `SendInput`) to navigate between relevant work windows.
4. **Desktop Notifications**: Pushes native Windows Action Center notifications (`Windows.UI.Notifications`).

---

## 3. The Real Snapdragon Advantage

To win a competition, NEXUS must clearly demonstrate advantages that **cannot be replicated on traditional x86 laptops**:

```
+---------------------------+-------------------------------+-------------------------------+
| Metric                    | Traditional x86 Laptop        | NEXUS on Snapdragon X         |
|                           | (Intel Core Ultra / AMD + dGPU)| (Copilot+ PC / Hexagon NPU)   |
+---------------------------+-------------------------------+-------------------------------+
| NPU Compute Dedicated     | 10 - 16 TOPS (Legacy)         | 45 TOPS (Hexagon Dedicated)   |
| Total Workload Power Draw | 35W - 75W (Fans spin loud)    | 3W - 7W (Completely silent)   |
| Battery Drain on Continuous| 2.0 - 3.5 hours               | 12.0 - 16.0 hours             |
| CPU Core Availability     | 60-80% CPU Load (System lags) | < 5% CPU Load (12 Oryon cores |
|                           |                               |   completely free for apps)   |
| Memory Architecture       | Split RAM + VRAM (PCIe copy)  | Unified LPDDR5x (135.6 GB/s)  |
| Resident Model RAM        | 4GB - 8GB                     | 1.89 GB (w4a16 / w8a16)       |
| Privacy & Air-Gap         | Often requires Cloud API      | 100% Offline / Zero Network   |
+---------------------------+-------------------------------+-------------------------------+
```

### Key Demonstrable Differentiators:
1. **Zero CPU/GPU Interference**: While NEXUS transcribes meetings and indexes the screen at 45 TOPS, the user can compile code, render 4K video, or play games on the Oryon CPU and Adreno GPU without experiencing frame drops or system hitching.
2. **Thermal & Battery Invariance**: Running continuous local multimodal inference on x86 drains laptops within 2 hours. On Snapdragon X, the Hexagon NPU executes matrix computations at milliwatt efficiency, enabling an **all-day ambient assistant**.
3. **Instant Responsiveness**: The unified memory (135 GB/s) and pre-compiled QNN context binaries eliminate PCIe bus latency and JIT compile freezes.

---

## 4. Software Stack & Hardware Boundaries

### 4.1 Dependency Stack
- **OS**: Windows 11 on ARM64 (Build 26100.x or newer).
- **Driver**: Qualcomm Hexagon NPU Compute Driver (`v1.0.0.10+`).
- **Language**: Python 3.10+ (Native ARM64 build on target PC) or C++20 / C# for native shell.
- **Inference Runtimes**:
  - `onnxruntime-qnn` (ARM64 wheel with `QnnHtp.dll` backend).
  - `onnxruntime-genai` (ARM64 with QNN execution provider support).
- **Model Suite**:
  - `Whisper-Small-Quantized` (`w8a16`, QNN Context Binary).
  - `EasyOCR` (`w8a16`, QNN ONNX).
  - `OpenAI-Clip` (`w8a16`, QNN Context Binary).
  - `all-MiniLM-L6-v2` (`w8a16`, QNN Context Binary).
  - `Llama-v3.2-1B-Instruct` (`w4a16`, EPContext QNN GenAI).

### 4.2 Cross-Platform Development & Dual-Mode Engine Architecture

Because the active development environment is an x64 Windows workstation, the codebase must be designed with an abstraction layer:

```python
# Conceptual Runtime Dispatcher in NEXUS
class NexusInferenceEngine:
    def __init__(self, mode="auto"):
        self.device = self._detect_platform(mode)
        
    def _detect_platform(self, mode):
        import platform
        arch = platform.machine().lower()
        if mode == "snapdragon-npu" or (mode == "auto" and arch in ["arm64", "aarch64"]):
            return {
                "provider": "QNNExecutionProvider",
                "options": {
                    "backend_path": "QnnHtp.dll",
                    "htp_performance_mode": "sustained_high_performance",
                    "context_cache_enable": 1
                }
            }
        else:
            # Fallback for x64 development workstation
            return {
                "provider": "CPUExecutionProvider",
                "options": {}
            }
```

This ensures that:
- **Development, prompt iteration, UI design, and test suites** run immediately on the x64 development machine.
- **Compilation and profiling** are submitted to the Qualcomm AI Hub cloud farm targeting the "Snapdragon X Elite CRD".
- **Deployment and competition demonstration** run on physical Snapdragon X hardware with 100% NPU acceleration.

---

## 5. Technical Risk Analysis & Mitigation

| Identified Risk | Severity | Mitigation Strategy |
| :--- | :--- | :--- |
| **NPU Context Compilation Failure** | High | Pre-compile all graphs offline using Qualcomm AI Hub into static `.bin` context files; never use JIT compilation on device. |
| **NPU Memory Allocation Contention** | Medium | Sequence pipeline execution: ASR and OCR execute prior to SLM inference; total concurrent model footprint kept strictly under 2.0 GB. |
| **SLM Context Window Truncation** | Medium | Fixed 2048-token context window on NPU; enforce semantic chunk filtering in MiniLM retrieval layer to pass only the top 3 relevant context paragraphs. |
| **x64 Developer Machine Divergence** | Low | Implement clean runtime adapter pattern (`QNNExecutionProvider` on ARM64 vs `CPUExecutionProvider` on x64). |
