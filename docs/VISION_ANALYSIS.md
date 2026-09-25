# NEXUS: Multimodal Vision & Image Understanding Architecture (Phase 9)

**Document**: `docs/VISION_ANALYSIS.md`  
**Target Platform**: Qualcomm Snapdragon X Elite / Plus (Windows 11 ARM64) & Windows AMD64 Host  
**Primary Vision Model**: `ResNet-18` (Dual Output: 1000-class ImageNet Classification + 512-dim Normalized Feature Embedding)  
**Model Source**: PyTorch / TorchVision (`ResNet18_Weights.DEFAULT`), exported to standalone ONNX (Opset 18) at `models/vision/resnet18_vision.onnx`  
**Runtime**: ONNX Runtime (`CPUExecutionProvider` [VERIFIED]; `QNNExecutionProvider` [SUPPORTED BY CODE / NOT PHYSICALLY VERIFIED])  
**Pipeline Integration**: Composite Perception Architecture (`ResNet-18` + `EasyOCR` / `Tesseract` + `Llama-v3.2-1B-Instruct`)  

---

## 1. Executive Summary & Vision Philosophy

In enterprise, engineering, and technical document analysis, multimodal comprehension requires rigorous epistemic precision:
1. **Never Confuse Measurement with Deduction**: Optical measurements directly extracted from pixels (**`OBSERVED`**) must be strictly distinguished from semantic interpretations and statistical classifications (**`INFERRED`**).
2. **Never Claim Unsupported Capabilities**: `ResNet-18` is a genuine 1000-class ImageNet convolutional neural network generating 512-dimensional visual feature vectors. It is not an industrial defect microscope: microscopic surface fatigue cracks, metallurgical pitting, or tiny alphanumeric strings require dedicated inspection models or dedicated OCR backbones.
3. **Broad Document & Image Format Support**:
   - Native images: **PNG**, **JPG**, **JPEG**.
   - **Images embedded in PDFs**: Extracted directly from PDF page streams without external cloud converters.
4. **Zero Unrestricted Camera Monitoring**: All image understanding operates strictly on-demand against user-selected local documents. No continuous webcam streaming, background frame polling, or unapproved surveillance.

---

## 2. Epistemic Separation: OBSERVED vs INFERRED

```
+-------------------------------------------------------------------------+
|                              IMAGE INPUT                                |
|                        (PNG, JPG, or PDF Image)                         |
+------------------------------------+------------------------------------+
                                     |
         +---------------------------+---------------------------+
         |                                                       |
         v                                                       v
+----------------------------------+   +----------------------------------+
|             OBSERVED             |   |             INFERRED             |
|   (Measurable Optical Facts)     |   |   (Semantic Deductions & Tags)   |
+----------------------------------+   +----------------------------------+
| • Exact Dimensions: 1920x1080    |   | • Visual Category: Technical     |
| • Aspect Ratio: 1.78 (16:9)      |   |   Diagram / Schematic            |
| • Color Mode: RGB (3-channel)    |   | • Semantic Tags: [chart, blue,   |
| • Channel Statistics (Mean, Std) |   |   circuit, industrial]           |
| • Mean Luminance: 142.6 / 255    |   | • High-level Description         |
| • Contrast: 68.4 RMS             |   | • Classification Confidence: 92% |
| • Dominant RGB Color Palette     |   | • Multimodal Reasoning Summary   |
| • Visual Complexity Entropy      |   | • Zero-shot Similarity Scores    |
+----------------------------------+   +----------------------------------+
```

### 2.1 The `OBSERVED` Block
Measurable physical and mathematical attributes computed deterministically from the pixel grid:
- `dimensions`: Width and height in pixels.
- `aspect_ratio`: $\text{width} / \text{height}$.
- `format`: Concrete file format (`PNG`, `JPEG`).
- `color_mode`: Color space (`RGB`, `RGBA`, `L`).
- `channel_stats`: Mean and standard deviation per color channel.
- `brightness`: Perceived luminance ($0.299R + 0.587G + 0.114B$).
- `contrast`: Standard deviation of luminance.
- `dominant_palette`: Top RGB and hex color clusters.
- `complexity_entropy`: Shannon entropy representing visual texture density.

### 2.2 The `INFERRED` Block
High-level deductions produced by the visual encoder (`OpenAI-Clip`) and multimodal classifier:
- `visual_category`: Scene/document taxonomy (e.g. `technical_diagram`, `dashboard_screenshot`, `document_page`, `photograph`, `data_visualization`).
- `semantic_tags`: Top ranking visual semantic labels.
- `description`: Structured natural language interpretation of the visual scene.
- `confidence`: Calibrated classification confidence ($0.0$ to $1.0$).
- `capabilities_boundary`: Explicit notice detailing what was measured vs deduced.

---

## 3. Format Support & PDF Image Extraction

NEXUS handles three primary visual inputs:

1. **Standalone PNG Images**: Lossless technical schematics, UI screenshots, diagrams.
2. **Standalone JPG / JPEG Images**: Field photos, inspection captures, scanned records.
3. **Images Extracted from PDFs**:
   - Uses `pypdf` page image streams to extract embedded raster images directly in-memory.
   - Preserves page provenance (`page_number`, image index, dimensions).
   - Allows analyzing embedded photos and technical diagrams inside multi-page reports.

---

## 4. Hardware Optimization & Snapdragon Execution

- **Target Silicon**: Qualcomm Hexagon NPU on Snapdragon X Elite / Plus (`QnnHtp.dll` on Windows 11 ARM64).
- **CPU Execution Provider**: `CPUExecutionProvider` [**VERIFIED**]
  - Multi-threaded AVX2/NEON ONNX Runtime kernel execution.
  - Generates genuine 1000-class ImageNet distributions and 512-dim visual embeddings locally.
- **Snapdragon NPU Execution Provider**: `QNNExecutionProvider` (`QnnHtp.dll`) [**SUPPORTED BY CODE** / **QUALCOMM-DOCUMENTED**]
  - Integrated via ONNX Runtime QNN Execution Provider.
  - Automatically detected on ARM64 Windows 11 with Qualcomm AI Engine Direct runtime.
- **Physical Snapdragon NPU Execution**: [**NOT PHYSICALLY VERIFIED**]
  - Development host is AMD64 Windows; physical Hexagon NPU execution remains unverified until deployed on physical Snapdragon silicon.
- **Hardware Blocker Logging**:
  - Running on non-ARM64 hosts surfaces explicit blocker diagnostics in `/vision/status` rather than simulating fake NPU status.

---

## 5. Security & Privacy Safeguards

- **Zero Camera Monitoring**: No background video feeds, no periodic webcam frame capture, no camera daemon.
- **Strict Path Sandboxing**: Analyzes only files stored in `data/documents/<uuid>/`. Rejects path traversal and external URI injection.
