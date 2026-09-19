# NEXUS: Technical Feasibility Analysis & Qualcomm AI Ecosystem Evaluation

**Project**: NEXUS — Offline-First Multimodal AI Work Agent for Snapdragon-Powered Windows PCs  
**Status**: Technical Research & Feasibility Phase  
**Target Hardware**: Snapdragon X Elite, Snapdragon X Plus, Snapdragon X2 Series  
**Target OS**: Windows 11 on ARM64 (Build 24H2+ / Copilot+ PC Baseline)

---

## 1. Qualcomm AI Hub & Developer Ecosystem

The Qualcomm AI Hub (`aihub.qualcomm.com`) is Qualcomm's managed developer platform and model repository designed to optimize, compile, and validate machine learning models specifically for Qualcomm hardware (Hexagon NPU, Adreno GPU, and Oryon CPU).

### 1.1 Architecture & Services
1. **Model Catalog & Repository (`qai-hub-models`)**:
   - Curated, pre-optimized implementations of popular vision, audio, embedding, and generative architectures.
   - Hosted publicly on GitHub (`qualcomm/ai-hub-models`) and Hugging Face (`huggingface.co/qualcomm`).
   - Models come with target-specific optimization scripts, quantization recipes (using AIMET - AI Model Efficiency Toolkit), and serialization configurations.
2. **Cloud Compilation Service**:
   - Developers submit PyTorch, ONNX, or TensorFlow models via the Python SDK (`qai-hub`) or CLI.
   - The cloud backend invokes the target hardware's compiler toolchain (part of Qualcomm AI Engine Direct / QNN SDK) without requiring the developer to configure cross-compilation toolchains locally.
   - Produces serialized target artifacts: **QNN Context Binaries (`.bin`)**, **QNN Precompiled ONNX (`EPContext` nodes)**, or **TFLite flatbuffers**.
3. **Cloud Device Farm & Profiler**:
   - Qualcomm maintains hosted physical hardware racks containing **Snapdragon X Elite CRD (Compute Reference Design)** boards and commercial Copilot+ PCs.
   - Allows profiling models on physical silicon directly from the cloud: measuring layer-by-layer execution time, memory bandwidth, peak VRAM/RAM allocation, and compute offload percentage (% NPU vs CPU vs GPU).
4. **Local Tooling**:
   - Python library `qai-hub` and `qai-hub-models`.
   - Qualcomm AI Engine Direct SDK (QAIRT / QNN SDK) available for native Windows on ARM64 development.

---

## 2. Snapdragon Hardware Platform Analysis

| Architectural Parameter | Snapdragon X Elite (e.g., X1E-84-100 / X1E-80-100) | Snapdragon X Plus (e.g., X1P-64-100 / X1P-42-100) | Snapdragon X2 Series (X2 Elite / Extreme / Plus) |
| :--- | :--- | :--- | :--- |
| **NPU Architecture** | Qualcomm Hexagon HTP (v75 / v73 class) | Qualcomm Hexagon HTP (v75 / v73 class) | Next-Gen Qualcomm Hexagon HTP (v79+ class) |
| **NPU Compute** | **45 TOPS** (Dedicated NPU) | **45 TOPS** (Dedicated NPU) | **80 – 85 TOPS** (Dedicated NPU) |
| **CPU Architecture** | 12-core Qualcomm Oryon (ARMv8.7-A 64-bit) | 10-core or 8-core Qualcomm Oryon | Up to 18-core Qualcomm Oryon Gen 2 |
| **CPU Peak Clock** | Up to 3.8 GHz (4.2 GHz dual-core boost) | Up to 3.4 GHz | Up to 5.0 GHz |
| **GPU Architecture** | Qualcomm Adreno (3.8 – 4.6 TFLOPS) | Qualcomm Adreno (1.7 – 3.8 TFLOPS) | Up to 6.5+ TFLOPS Adreno Gen 2 |
| **Memory Architecture** | LPDDR5x (8448 MT/s), 16GB – 64GB unified | LPDDR5x (8448 MT/s), 16GB – 32GB unified | LPDDR5x (9600+ MT/s), up to 128GB unified |
| **Memory Bandwidth** | **135.6 GB/s** (Unified zero-copy) | **135.6 GB/s** (Unified zero-copy) | **~153+ GB/s** (Unified zero-copy) |
| **Always-On Sensing** | Dual Micro NPU in Sensing Hub | Dual Micro NPU in Sensing Hub | Enhanced Multi-Micro NPU Sensing Hub |
| **Process Node** | 4nm TSMC | 4nm TSMC | 3nm TSMC |
| **Platform Availability** | In Production (Commercial Copilot+ PCs) | In Production (Commercial Copilot+ PCs) | Rolling out in Flagship Laptops / Mini-PCs |

### 2.1 Critical Hardware Insights for NEXUS
1. **NPU Parity Across Elite and Plus**: The dedicated Hexagon NPU delivers **identical 45 TOPS performance on both Snapdragon X Elite and Snapdragon X Plus**. This means NPU-compiled workloads (audio transcription, embeddings, OCR, vision encoding) will run at the exact same latency and throughput regardless of whether the user possesses an Elite or Plus laptop.
2. **Unified Memory Architecture**: Snapdragon X platforms feature a unified LPDDR5x memory architecture (135.6 GB/s). Unlike discrete desktop GPUs where weights must be copied across a PCIe bus (introducing 10–50ms transfers), memory is shared between the Oryon CPU, Adreno GPU, and Hexagon NPU. However, the NPU uses physical TCM (Tightly Coupled Memory) and contiguous DMA buffers, meaning memory allocation must still be managed carefully.
3. **Snapdragon X2 Forward Compatibility**: Models compiled for the Hexagon HTP architecture on Snapdragon X series will run seamlessly on Snapdragon X2, benefiting from roughly double the matrix multiply throughput (80–85 TOPS) and higher memory bandwidth without structural model rework.

---

## 3. Windows on Snapdragon (WoS) Operating System Stack

Windows 11 version 24H2 is the foundational OS for Copilot+ PCs. It incorporates critical kernel, driver, and runtime subsystems tailored for Snapdragon ARM64:

1. **Native Execution vs. Prism Emulation**:
   - NEXUS **must** run as a native ARM64 or ARM64EC application. Running an AI agent under x64 Prism emulation incurs translation overhead, prevents direct access to native ARM64 hardware drivers, and degrades memory throughput.
2. **Microsoft Compute Driver Model (MCDM)**:
   - Qualcomm provides an MCDM-compliant driver (`qcnpu*.sys`) that exposes the Hexagon NPU directly to Windows as a dedicated compute engine visible in Task Manager under the "NPU" tab.
3. **Driver Minimums**:
   - Qualcomm Hexagon NPU Driver: **v1.0.0.10** or newer.
   - Windows 11 Build: **26100 (24H2)** or newer.

---

## 4. Qualcomm-Supported Inference Runtimes

There are four primary runtimes capable of executing models on Snapdragon-powered Windows PCs. Their feasibility, performance, and maturity are evaluated below:

```
+---------------------------------------------------------------------------------------+
|                                     NEXUS Application                                 |
+---------------------------------------------------------------------------------------+
        |                                       |                               |
        v                                       v                               v
+-----------------------+           +-----------------------+           +---------------+
|  ONNX Runtime (QNN)   |           |  ONNX Runtime GenAI   |           |  DirectML EP  |
|  - QnnHtp.dll         |           |  - QNN EP Context     |           |  - Direct3D12 |
|  - Precompiled .bin   |           |  - KV-cache on NPU    |           |  - NPU / GPU  |
+-----------------------+           +-----------------------+           +---------------+
        |                                       |                               |
        +-------------------+-------------------+                               |
                            v                                                   v
        +---------------------------------------+               +-----------------------+
        |  Qualcomm AI Engine Direct (QAIRT)    |               |  Microsoft MCDM       |
        |  Hexagon Tensor Processor (HTP v75)   |               |  Device Driver        |
        +---------------------------------------+               +-----------------------+
                            |                                                   |
                            +-------------------+-------------------------------+
                                                v
                                +-------------------------------+
                                |  Snapdragon X Hexagon NPU     |
                                |  (45 TOPS Dedicated Silicon)  |
                                +-------------------------------+
```

### 4.1 ONNX Runtime with QNN Execution Provider (`onnxruntime-qnn`)
- **Feasibility**: **HIGH / PRODUCTION-READY**
- **Mechanism**: ONNX Runtime interacts with the Qualcomm AI Engine Direct (QNN) SDK via `QNNExecutionProvider`. The provider loads `QnnHtp.dll` (Hexagon Tensor Processor backend).
- **Context Caching (`context_cache_enable`)**: Crucial optimization. When enabled, ORT bypasses on-device graph compilation and loads a pre-compiled `.bin` context binary directly into NPU memory in sub-100ms.
- **Suitability**: Ideal for static vision, audio, OCR, and embedding models (Whisper encoder/decoder, CLIP, MiniLM, EasyOCR).

### 4.2 ONNX Runtime GenAI with QNN Execution Provider (`onnxruntime-genai`)
- **Feasibility**: **MODERATE / VERIFIED**
- **Mechanism**: Microsoft and Qualcomm jointly maintain GenAI extensions for autoregressive text generation. It encapsulates model graph execution, multi-turn KV-caching, and token search into an ARM64 native binary targeting `QNNExecutionProvider`.
- **Model Packaging**: Requires an ONNX wrapper with embedded or external `EPContext` nodes referencing the compiled HTP context binary, along with `genai_config.json`, `tokenizer.json`, and quantization nodes.
- **Suitability**: Essential for running Small Language Models (Llama-3.2-1B, Phi-3.5-mini) on the NPU.

### 4.3 DirectML Execution Provider (`DmlExecutionProvider`)
- **Feasibility**: **HIGH (Fallback & Hybrid)**
- **Mechanism**: Microsoft DirectML targets hardware via DirectX 12. On Snapdragon X, DirectML supports both the Adreno GPU and (via MCDM driver) the Hexagon NPU.
- **Trade-off**: DirectML NPU operator coverage is narrower than QNN native, and certain complex custom ops silently fall back to CPU. However, DirectML on the **Adreno GPU** is extremely mature and provides a rock-solid fallback when NPU operator graphs fail.

### 4.4 `llama.cpp` / GGUF on Snapdragon
- **Feasibility on NPU**: **LOW / EXPERIMENTAL**
- **Investigation Finding**: Upstream `llama.cpp` does **NOT** support a stable QNN NPU backend out-of-the-box. GGUF relies on dynamic quantization and tensor execution that conflicts with the Hexagon NPU's requirement for static graph topologies and compiled context binaries.
- **Feasibility on CPU / GPU**: **HIGH** — Upstream `llama.cpp` runs natively on Oryon CPU via ARM64 NEON, and Qualcomm maintains an official OpenCL backend for Adreno GPU.
- **Verdict for NEXUS**: For pure NPU execution, `onnxruntime-genai` + QNN EP is required. `llama.cpp` should only be considered as a CPU/GPU auxiliary runtime.

---

## 5. NPU Execution Specifics: Constraints & Mechanics

The Hexagon NPU is not a general-purpose processor; it is a specialized SIMD/VLIW matrix and tensor accelerator consisting of **Hexagon Vector eXtensions (HVX)** and **Hexagon Matrix eXtensions (HMX)**.

### 5.1 Strict Hardware Requirements
1. **Mandatory Quantization**:
   - The Hexagon Tensor Processor (HTP) operates at peak 45 TOPS efficiency in **INT8** (weights) / **INT8** (activations) or **INT4** (weights) / **INT16** (activations) mode (`w4a16`, `w8a16`, `w8a8`).
   - While HTP v73/v75 has limited FP16 support, standard **FP32 execution is completely unsupported on the NPU**. An unquantized FP32 model will either be rejected at compilation or partitioned with layers falling back to the CPU.
2. **Static Dimension Requirements**:
   - NPU context binaries require static tensor shapes (batch size, sequence length, image height/width). Dynamic shapes force on-the-fly graph recompilation or CPU fallback.
3. **Graph Partitioning & Fallback Penalty**:
   - If a single operator in an ONNX model is unsupported by the QNN HTP backend, ORT partitions the graph. The intermediate tensor must be copied from NPU TCM across memory to CPU cache, evaluated on CPU, and copied back to NPU. This memory synchronization round-trip obliterates latency advantages.
4. **Context Binaries (`.bin`)**:
   - Production deployment mandates pre-generating QNN Context Binaries offline. Launching an uncompiled ONNX model directly against `QNNExecutionProvider` on-device triggers JIT compilation that can freeze the application for 1–5 minutes on first launch.

---

## 6. End-to-End Model Deployment Workflow

```
[PyTorch / Hugging Face Source]
               |
               v
[Quantization: AIMET / QNN Quantizer]  ---> (w8a16, w4a16, or int8)
               |
               v
[Qualcomm AI Hub Cloud Compiler]        ---> Target: "Snapdragon X Elite CRD"
               |                             Runtime: onnx / qnn_context_binary
               v
[Generated Artifacts]                   ---> 1. model_quantized.onnx (EPContext)
                                             2. qnn_context.bin (Serialized HTP graph)
                                             3. genai_config.json / tokenizer.json
               |
               v
[Bundled in NEXUS Desktop Package]     ---> Local disk (C:\Program Files\NEXUS\models\)
               |
               v
[Offline Execution via ORT / QNN EP]   ---> Loads QnnHtp.dll, passes .bin directly to NPU
```

---

## 7. Profiling & Performance Validation

### 7.1 Qualcomm AI Hub Cloud Profiling
- Submits candidate models to Qualcomm's hardware farm.
- Provides granular telemetry:
  - Total inference latency (median, 90th percentile).
  - Memory consumption (peak RAM).
  - Compute unit utilization breakdown: NPU vs. CPU vs. GPU.
  - Layer-by-layer compute time and identifying fallback nodes.

### 7.2 On-Device Profiling (Local)
- **QNN Profiler**: Executing using `qnn-net-run` with `--profiling_level detailed` outputs `qnn-profiling-data.log`. Parsed via `qnn-profile-viewer` to inspect hardware cycle counts on HTP.
- **Windows Task Manager & WPA**: Using Windows Performance Analyzer with Microsoft-Windows-DxgKrnl providers verifies that the workload appears under the "Compute 0 (NPU)" engine rather than consuming CPU cores.

---

## 8. Offline Inference Feasibility

A core tenant of NEXUS is **100% offline-first execution**:

1. **Compilation vs. Runtime Separation**:
   - Qualcomm AI Hub is an **ahead-of-time (AOT) tool**. It is used during development to prepare, quantize, and compile models.
   - At runtime on the user's PC, **zero network access is required**. All model weights, context binaries, tokenizers, and libraries are stored locally on the SSD.
2. **Local Memory Footprint Feasibility**:
   - A standard Snapdragon X Copilot+ PC comes with 16GB or 32GB of unified LPDDR5x RAM.
   - Target model bundle footprint:
     - Whisper-Small Quantized: ~250 MB
     - EasyOCR / TrOCR: ~150 MB
     - all-MiniLM-L6-v2: ~80 MB
     - CLIP Vision Encoder: ~300 MB
     - Llama-3.2-1B-Instruct (w4a16): ~1.2 GB
     - **Total Model Resident Footprint**: **~2.0 GB**
   - 2.0 GB represents only **12.5%** of a 16GB machine's unified memory, leaving over 10GB for Windows OS, desktop apps, and active user workflows.

---

## 9. Development Environment Findings & Architectural Impact

An audit of the current developer workstation reveals:
- **Host Architecture**: `x64-based PC` (Windows 11 Home, Intel/AMD).
- **Target Architecture**: `ARM64-based Snapdragon X Elite / Plus PC`.

### Strategic Feasibility Consequence:
Because the development host is an x64 PC, native Hexagon NPU libraries (`QnnHtp.dll` for ARM64) **cannot be executed locally on this workstation**. Attempting to load `QnnHtp.dll` on an x64 machine will result in architecture mismatch errors (`%1 is not a valid Win32 application`).

**Solution**:
1. Leverage **Qualcomm AI Hub Cloud Services (`qai-hub`)** to compile and profile models on physical Snapdragon X hardware hosted in Qualcomm's cloud farm.
2. Structure the NEXUS runtime engine with a **Dual-Mode Backend Architecture**:
   - **Target Mode (`snapdragon-npu`)**: Loads `QNNExecutionProvider` with `backend_path="QnnHtp.dll"` for deployment on Snapdragon X PCs.
   - **Local Simulation / Development Mode (`cpu-fallback`)**: Loads standard `CPUExecutionProvider` or `DmlExecutionProvider` on x64 development machines to verify functional logic, prompts, agents, and pipelines without requiring physical Snapdragon hardware at every commit.
