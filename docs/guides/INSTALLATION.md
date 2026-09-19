# NEXUS Installation Guide

**Supported Platforms**:
- Windows 11 ARM64 (Snapdragon X Elite / X Plus) — Full NPU acceleration
- Windows 11 x64 (Intel / AMD) — Development / CPU fallback mode

---

## Prerequisites

### All Platforms

| Requirement | Minimum | Recommended |
|---|---|---|
| Python | 3.10 | 3.11 ARM64 native |
| RAM | 8 GB | 16 GB |
| Disk | 5 GB free | 10 GB free |
| OS | Windows 11 22H2 | Windows 11 24H2+ |

### Snapdragon-Specific

| Requirement | Details |
|---|---|
| Qualcomm Hexagon Driver | `QnnHtp.dll` v1.0.0.10 or later |
| onnxruntime-qnn | ARM64 wheel with QNN Execution Provider |
| Windows on ARM | Build 26100 or newer |

---

## Step 1: Clone the Repository

```bash
git clone https://github.com/your-org/nexus.git
cd nexus
```

---

## Step 2: Create a Virtual Environment

```bash
# Windows (PowerShell)
python -m venv .venv
.venv\Scripts\Activate.ps1

# macOS / Linux (development only)
python -m venv .venv
source .venv/bin/activate
```

---

## Step 3: Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

> [!NOTE]
> On Snapdragon ARM64, ensure you are using an ARM64-native Python build.
> Check with: `python -c "import platform; print(platform.machine())"`
> Expected output: `ARM64` or `aarch64`

---

## Step 4: Verify Installation

```bash
python -c "import onnxruntime as ort; print('Providers:', ort.get_available_providers())"
```

Expected on **Snapdragon**:
```
Providers: ['QNNExecutionProvider', 'CPUExecutionProvider']
```

Expected on **x64 development host**:
```
Providers: ['AzureExecutionProvider', 'CPUExecutionProvider']
```

---

## Step 5: Initialize Data Directories

The backend creates these automatically on first run, but you can create them manually:

```bash
mkdir -p data/documents data/reports data/audio
```

---

## Step 6: Generate Demo Data

```bash
python scripts/generate_demo_data.py
```

This creates the synthetic industrial inspection dataset in `demo_data/`.

---

## Step 7: Start the Backend

```bash
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

You should see:
```
INFO: Starting NEXUS v1.0.0 [production]
INFO: Target Platform: snapdragon-x-elite | Offline Mode: True
INFO: Application startup complete.
```

---

## Step 8: Open the UI

Navigate to `http://127.0.0.1:8000` in your browser.

---

## Step 9: Run the Test Suite

```bash
pytest tests/ -v
```

All 133+ tests should pass on both ARM64 and x64.

---

## Step 10: Run the Competition Demo

```bash
python demo.py
```

---

## Troubleshooting

See [`docs/guides/TROUBLESHOOTING.md`](TROUBLESHOOTING.md) for common issues.
