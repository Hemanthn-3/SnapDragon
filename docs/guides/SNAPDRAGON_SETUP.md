# NEXUS Snapdragon Setup Guide

**Target Hardware**: Snapdragon X Elite, Snapdragon X Plus, Snapdragon X2  
**Target OS**: Windows 11 ARM64 (Build 26100+, Copilot+ PC)

---

## 1. Verify Snapdragon Hardware

Open PowerShell and confirm ARM64:

```powershell
(Get-WmiObject -Class Win32_Processor).Architecture
# Should return: 12 (ARM64)

python -c "import platform; print(platform.machine())"
# Should return: ARM64
```

---

## 2. Install the Qualcomm Hexagon Driver

The Qualcomm Hexagon NPU Compute Driver (`QnnHtp.dll`) is required for NPU acceleration.

### Automatic (Recommended)

On genuine Snapdragon X Copilot+ PCs, the driver is pre-installed or available through Windows Update:

1. Open **Settings** → **Windows Update** → **Advanced options** → **Optional updates**
2. Install any **Qualcomm** driver updates
3. Restart the PC

### Manual Installation

Download from Qualcomm's developer portal or your PC manufacturer's support page:

```
Driver package: Qualcomm AI Stack (QAS) for Windows on ARM
Minimum version: QnnHtp v1.0.0.10
```

Verify driver installation:
```powershell
Get-WmiObject Win32_PnPSignedDriver | Where-Object { $_.DeviceName -like "*Hexagon*" }
```

---

## 3. Install ARM64 Python

Download the Python 3.11 **ARM64** installer from python.org:

```
https://www.python.org/downloads/release/python-3119/
→ Windows installer (ARM64)
```

Verify:
```powershell
python -c "import platform; print(platform.machine(), platform.python_implementation())"
# Expected: ARM64 CPython
```

---

## 4. Install onnxruntime-qnn

The standard `onnxruntime` package on PyPI is x64. For Snapdragon, install the QNN-enabled ARM64 wheel:

```bash
# Install onnxruntime with QNN execution provider (ARM64)
pip install onnxruntime-qnn

# OR install from Qualcomm's release:
pip install https://github.com/microsoft/onnxruntime/releases/download/v1.20.0/onnxruntime_qnn-1.20.0-cp311-cp311-win_arm64.whl
```

Verify QNN is available:
```python
import onnxruntime as ort
providers = ort.get_available_providers()
print("QNN available:", "QNNExecutionProvider" in providers)
# Expected: QNN available: True
```

---

## 5. Install onnxruntime-genai (for LLM)

```bash
pip install onnxruntime-genai-qnn
```

Or build from source if the pre-built wheel is unavailable:
```bash
git clone https://github.com/microsoft/onnxruntime-genai
cd onnxruntime-genai
python build.py --config Release --use_qnn
```

---

## 6. Install NEXUS Dependencies

```bash
cd nexus/
pip install -r requirements.txt
```

> [!NOTE]
> `requirements.txt` includes the standard `onnxruntime` as a fallback for x64 development.
> On ARM64, the `onnxruntime-qnn` wheel you installed in Step 4 takes precedence.

---

## 7. Download Model Assets

NEXUS models must be downloaded before the first run. Use Qualcomm AI Hub:

```bash
# Install qai-hub-models
pip install qai-hub-models

# Authenticate (requires free Qualcomm AI Hub account)
qai-hub configure --api_token YOUR_API_TOKEN

# Download quantized models
qai-hub-models fetch Whisper-Small-Quantized --runtime qnn_context_binary --precision w8a16
qai-hub-models fetch all-minilm-l6-v2 --runtime qnn_context_binary --precision w8a16
qai-hub-models fetch OpenAI-Clip --runtime qnn_context_binary --precision w8a16
```

For Llama-3.2-1B, use Hugging Face:
```bash
pip install huggingface_hub
huggingface-cli download qualcomm/Llama-v3.2-1B-Instruct --local-dir models/llm/
```

See [`docs/guides/MODEL_SETUP.md`](MODEL_SETUP.md) for full model setup instructions.

---

## 8. Verify NPU Activation

Start NEXUS and check the health endpoint:

```bash
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
curl http://127.0.0.1:8000/health
```

In the response, look for:
```json
{
  "npu_available": true,
  "execution_provider": "QNNExecutionProvider",
  "platform": "Snapdragon X Elite"
}
```

The NEXUS UI Home screen will show **NPU: ACTIVE** in the system status bar when `QNNExecutionProvider` is detected.

---

## 9. Run the Demo

```bash
python demo.py
```

The NPU badge in the demo's hardware status block will show **ACTIVE** (green) if the Hexagon NPU is operational.

---

## Performance Expectations on Snapdragon X Elite

Based on Qualcomm AI Hub specifications (not measured on this device):

| Component | Expected Latency | vs. CPU Baseline |
|---|---|---|
| ASR (Whisper-Small, 5s audio) | < 120 ms | ~4x faster |
| Embedding (MiniLM, 128 tokens) | < 15 ms | ~3x faster |
| Vision (CLIP, 224×224) | < 35 ms | ~1.5x faster |
| LLM (Llama-3.2-1B, 16 tokens) | < 400 ms | ~2x faster |
| Token throughput | 40–48 tokens/sec | ~3x faster |

> [!IMPORTANT]
> These figures are from Qualcomm AI Hub documentation and represent expected performance. Actual measurements on specific hardware may vary. NEXUS reports actual measured values — not claimed values.

---

## Troubleshooting Snapdragon-Specific Issues

**QNNExecutionProvider not in providers list**
- Confirm `onnxruntime-qnn` is installed (not `onnxruntime`)
- Confirm Python process is ARM64 native (not x64 emulated)
- Confirm Qualcomm Hexagon driver is installed and updated

**Model loading fails with QNN errors**
- Ensure QNN context binary matches your driver version
- Try CPU fallback: `NEXUS_PROVIDER=cpu python -m uvicorn backend.main:app ...`

**Memory errors on 8GB device**
- NEXUS full pipeline peak RSS is ~1.89 GB
- Close other heavy applications before running
- Use `--workers 1` with uvicorn to prevent duplicate model loads
