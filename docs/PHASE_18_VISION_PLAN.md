# NEXUS — Phase 18 Vision Replacement Plan
## Replacing Fake/Deterministic Vision with Genuine Neural Network Inference

**Document Version:** 1.0.0  
**Phase:** 18  
**Status:** In Progress  
**Target:** Snapdragon X Elite (ARM64 Windows 11) & Windows AMD64 Development Host  

---

## 1. Current Implementation Inspection

The current vision adapter in [`backend/models_local/clip_vision.py`](file:///c:/Users/heman/Desktop/Snapdragon/backend/models_local/clip_vision.py) is completely synthetic and does not execute any neural network.

### Why It Is Not Genuine Neural Inference:
1. **Zero Model Weights Loaded**:
   In `load()`, the code merely calls `onnxruntime.get_available_providers()`, sets `self._execution_provider = "CPUExecutionProvider"`, and sets `self._status = ModelStatus.READY`. No ONNX model file, PyTorch model, or neural weights are ever loaded into device memory.
2. **Deterministic Downsampling as "Embedding"**:
   In `encode_image()`, the image pixels are flattened into an array, partitioned into 513 linear bins, and the bin means are returned as a "512-dim visual embedding". This is simple pixel-binning, not a neural embedding learned by a vision transformer or convolutional network.
3. **Hard-coded Heuristic Rules as "Vision Understanding"**:
   In `inspect_image()`, semantic classification (`visual_category`, `semantic_tags`) is produced via heuristic thresholding on brightness, contrast, and Shannon entropy:
   ```python
   if contrast < 25.0 and brightness > 220:
       visual_category = "document_page"
   elif entropy < 3.5 and len(dominant_palette) <= 3:
       visual_category = "technical_diagram"
   elif brightness < 80.0:
       visual_category = "dark_mode_ui_or_dashboard"
   ```
4. **False Claims of ViT-B/32 Quantization**:
   The metadata dictionary in `get_status()` claimed `OpenAI-CLIP-ViT-B32-Quantized` with `w8a16` quantization and `Snapdragon X Elite Hexagon NPU` target, despite no CLIP weights existing in `models/vision/OpenAI-Clip`.

---

## 2. Selected Replacement Model

### Model Architecture
- **Model**: `ResNet-18` (Dual Output: Classification Logits + 512-dim Feature Embedding)
- **Framework & Format**: ONNX (Open Neural Network Exchange), Opset 18
- **Model File**: `models/vision/resnet18_vision.onnx`
- **Model Source**: Pretrained weights from PyTorch / TorchVision (`ResNet18_Weights.DEFAULT`, trained on ImageNet-1k, 1000 categories).
- **Qualcomm AI Hub Alignment**: ResNet and MobileNet architectures are officially supported, profiled, and optimized by Qualcomm AI Hub for Snapdragon X Elite and Hexagon NPU inference via ONNX Runtime QNN Execution Provider (`QNNExecutionProvider`).

### Model Specifications
- **Input Tensor**:
  - Name: `input`
  - Shape: `(batch_size, 3, 224, 224)` (1, 3, 224, 224 for single-image inspection)
  - Data Type: `float32`
  - Preprocessing: Bilinear/bicubic resize to 224×224, normalized with standard ImageNet statistics:
    - Mean: `[0.485, 0.456, 0.406]`
    - Std: `[0.229, 0.224, 0.225]`
- **Output Tensors**:
  1. `logits`:
     - Shape: `(batch_size, 1000)`
     - Meaning: Unnormalized log-odds across 1000 ImageNet semantic classes.
     - Postprocessing: Softmax applied to derive probabilities and top-5 predicted visual categories.
  2. `embedding`:
     - Shape: `(batch_size, 512)`
     - Meaning: Visual feature vector extracted from the penultimate average pooling layer (`avgpool`).
     - Postprocessing: L2-normalized to produce unit-norm 512-dimensional visual embedding vectors.

---

## 3. Execution Pipeline

```
Raw Image Bytes (PNG / JPG / PDF Raster)
            ↓
PIL Decoding & Optical Measurements (Dimensions, Palette, Brightness, Entropy)
            ↓
Preprocessing: Resize (224x224) → Float32 → ImageNet Normalize → (1, 3, 224, 224)
            ↓
ONNX Runtime InferenceSession (QNNExecutionProvider on Snapdragon / CPUExecutionProvider Fallback)
            ↓
Model Outputs: Logits (1, 1000) & Embedding (1, 512)
            ↓
Postprocessing: Softmax → Top-5 Categories & L2 Normalized 512-dim Vector
            ↓
Strict Separation:
  - OBSERVED: Objective optical measurements (dimensions, aspect ratio, luminance, entropy, palette)
  - INFERRED: Neural classification predictions, top candidate labels, confidence scores, capabilities boundary
```

---

## 4. Execution Methods & Hardware Providers

### QNN Execution Method (Snapdragon X Elite)
- **Provider**: `QNNExecutionProvider`
- **Backend**: `QnnHtp.dll` (Qualcomm Hexagon Tensor Processor)
- **Target Hardware**: Snapdragon X Elite / Plus on Windows 11 ARM64
- **Verification Rule**: The system only reports `QNNExecutionProvider` if the session was successfully instantiated with the provider and verified via `session.get_providers()`.

### CPU Execution Method (Development & Fallback)
- **Provider**: `CPUExecutionProvider`
- **Backend**: ONNX Runtime CPU engine using multi-threaded AVX2/NEON kernels.
- **Guarantee**: Executes the **EXACT SAME** neural network ONNX graph as the NPU. Zero heuristic shortcuts or fake embeddings.

---

## 5. Physical Verification Status

| Component | Status | Truthful Reporting |
|---|---|---|
| ONNX Model Export & Verification | **VERIFIED** | Model exported to `models/vision/resnet18_vision.onnx` and verified |
| CPU Inference (`CPUExecutionProvider`) | **VERIFIED** | Real inference runs in ~10-25ms per image on CPU |
| Epistemic Demarcation | **VERIFIED** | Observed vs Inferred cleanly separated |
| Physical Snapdragon NPU (`QNNExecutionProvider`) | **SUPPORTED BY CODE / NOT PHYSICALLY VERIFIED** | Tested on AMD64 dev machine; requires physical Snapdragon X Elite device |

---

## 6. Known Capabilities & Boundaries

1. **General ImageNet Classification**:
   ResNet-18 is trained on ImageNet-1k (natural scenes, tools, vehicles, machines, equipment, structures). It provides genuine neural visual classification.
2. **Industrial Defect Boundaries**:
   General image classifiers can recognize machine assemblies, tools, fasteners, motors, and structures, but do not possess specialized microscopic metallurgical defect training. Claims of detecting microscopic fatigue cracks or surface pitting from generic models are explicitly disclaimed in `capabilities_boundary`.
3. **Text & Numbers**:
   Alphanumeric data within images must be extracted via the dedicated Tesseract OCR pipeline, not visual classification.
