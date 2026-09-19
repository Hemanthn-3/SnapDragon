# NEXUS Model Setup Guide

This guide explains how to obtain, configure, and verify all AI models used by NEXUS.

---

## Model Summary

| Role | Model | Source | Size | Format |
|---|---|---|---|---|
| ASR | Whisper-Small-Quantized | Qualcomm AI Hub | ~240 MB | QNN Context Binary |
| OCR | EasyOCR (CRAFT + CRNN) | Qualcomm AI Hub | ~120 MB | QNN ONNX |
| Vision | OpenAI-CLIP ViT-B/32 | Qualcomm AI Hub | ~300 MB | QNN Context Binary |
| Embedding | all-MiniLM-L6-v2 | Qualcomm AI Hub / HuggingFace | ~80 MB | ONNX (w8a16) |
| LLM | Llama-3.2-1B-Instruct | Qualcomm AI Hub / HuggingFace | ~1.15 GB | ORT GenAI + QNN |

**Total combined model footprint**: ~1.89 GB

---

## Option A: Qualcomm AI Hub (Snapdragon Target — Recommended)

Qualcomm AI Hub provides pre-compiled QNN context binaries optimized for Snapdragon X Elite.

### 1. Create a Free AI Hub Account

Register at: https://aihub.qualcomm.com

### 2. Install qai-hub-models

```bash
pip install qai-hub-models
qai-hub configure --api_token YOUR_API_TOKEN
```

### 3. Download Models

```bash
# ASR
qai-hub-models fetch Whisper-Small-Quantized \
  --runtime qnn_context_binary --precision w8a16 \
  --output-dir models/asr/

# Vision
qai-hub-models fetch OpenAI-Clip \
  --runtime qnn_context_binary --precision w8a16 \
  --output-dir models/vision/

# Embedding
qai-hub-models fetch all-minilm-l6-v2 \
  --runtime qnn_context_binary --precision w8a16 \
  --output-dir models/embedding/

# OCR
python -m qai_hub_models.models.easyocr.export \
  --target-runtime onnx \
  --device "Snapdragon X Elite CRD" \
  --output-dir models/ocr/
```

### 4. Download LLM

```bash
# Llama-3.2-1B-Instruct (QNN EPContext format)
python -m qai_hub_models.models.llama_v3_2_1b_instruct.export \
  --target-runtime onnx \
  --precision w4a16 \
  --output-dir models/llm/
```

---

## Option B: Hugging Face (Cross-Platform)

### Embedding Model (ONNX, CPU-compatible)

```bash
pip install huggingface_hub optimum
optimum-cli export onnx \
  --model sentence-transformers/all-MiniLM-L6-v2 \
  models/embedding/
```

### LLM (ONNX GenAI format)

```bash
huggingface-cli download qualcomm/Llama-v3.2-1B-Instruct \
  --local-dir models/llm/
```

### Whisper (CPU ONNX fallback)

```bash
pip install openai-whisper
python -c "import whisper; whisper.load_model('small')"
```

---

## Model Directory Structure

After setup, `models/` should look like:

```
models/
├── asr/
│   └── whisper_small_quantized/
│       ├── whisper_encoder.bin       (QNN context binary)
│       └── whisper_decoder.bin       (QNN context binary)
├── embedding/
│   └── all_minilm_l6_v2/
│       └── model.onnx                (ONNX w8a16)
├── vision/
│   └── clip_vit_b32/
│       ├── image_encoder.bin         (QNN context binary)
│       └── text_encoder.bin          (QNN context binary)
├── ocr/
│   └── easyocr/
│       ├── craft_text_detector.onnx
│       └── crnn_recognizer.onnx
└── llm/
    └── llama_3_2_1b_instruct/
        ├── model.onnx
        ├── model.onnx_data
        └── genai_config.json
```

---

## Configuring Model Paths

NEXUS automatically discovers models from the `models/` directory relative to the project root.

If your models are in a different location, set the environment variable:

```bash
# PowerShell
$env:NEXUS_MODEL_DIR = "C:\path\to\models"
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

---

## Verifying Models Load Correctly

Run the benchmark suite to confirm all models load:

```bash
python benchmarks/suite.py --iterations 1
```

Each component should show `status: VERIFIED` in the output. If a model file is missing, the component reports `status: NOT_FOUND` with the expected path.

---

## Development Mode (No Models Downloaded)

NEXUS runs without any model files on the development host. Components report their status honestly:

- ASR: `NOT_IMPLEMENTED` (no Whisper session)
- OCR: `NOT_YET_IMPLEMENTED` (no CRAFT+CRNN)
- Embedding: `VERIFIED` if `sentence-transformers` package is installed
- Vision: `VERIFIED` if CLIP ONNX model is present
- LLM: `VERIFIED` if Llama ONNX model is present

The competition demo and all API routes remain functional in development mode using the available components.

---

## License Notes

| Model | License |
|---|---|
| Whisper-Small | MIT |
| all-MiniLM-L6-v2 | Apache 2.0 |
| OpenAI-CLIP | MIT |
| Llama-3.2-1B-Instruct | Llama 3.2 Community License (Meta) |
| EasyOCR | Apache 2.0 |

Review all model licenses before distribution or commercial use.
