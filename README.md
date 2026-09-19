# NEXUS — Offline Multimodal Work Agent

**Snapdragon X Elite · Windows on ARM · 100% Air-Gapped · Zero Cloud**

[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-green)](https://fastapi.tiangolo.com)
[![ONNX Runtime](https://img.shields.io/badge/ONNX_Runtime-1.20+-orange)](https://onnxruntime.ai)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

---

## What Is NEXUS?

NEXUS is an autonomous, multimodal AI work agent engineered to run **entirely offline** on Qualcomm Snapdragon X Elite hardware. It ingests documents, images, audio, and structured data; reasons over them locally using a verified on-device model pipeline; and produces grounded, auditable work products — with **zero network calls, zero cloud APIs, and zero telemetry**.

Every cognitive stage runs on the **45 TOPS Qualcomm Hexagon NPU** via ONNX Runtime QNN Execution Provider, or gracefully degrades to CPU on development hosts.

```
User Goal ──► Plan ──► Document Discovery ──► OCR ──► Retrieval
                                                          │
                    Report ◄── Verification ◄── Evidence ◄── Vision + LLM
```

---

## Key Capabilities

| Capability | Technology | NPU-Accelerated |
|---|---|---|
| Voice command (ASR) | Whisper-Small-Quantized (w8a16) | ✅ Yes |
| Document text extraction (OCR) | EasyOCR CRAFT+CRNN (w8a8) | ✅ Yes |
| Visual scene analysis | OpenAI-CLIP ViT-B/32 (w8a16) | ✅ Yes |
| Semantic embeddings & retrieval | all-MiniLM-L6-v2 (w8a16) | ✅ Yes |
| Autonomous planning & reasoning | Llama-3.2-1B-Instruct (w4a16) | ✅ Yes |
| Grounded evidence verification | NEXUS Verifier (multi-layer) | CPU |
| Structured report generation | NEXUS Agent Orchestrator | CPU |

---

## Quick Start

### 1. Prerequisites

- Windows 11 ARM64 (Build 26100+) **or** Windows 11 x64 (development mode)
- Python 3.10 or 3.11 (ARM64 native recommended on Snapdragon)
- 8 GB RAM minimum (16 GB recommended)
- Qualcomm Hexagon NPU compute driver (for NPU acceleration)

### 2. Install

```bash
git clone https://github.com/your-org/nexus.git
cd nexus
pip install -r requirements.txt
```

### 3. Run the backend

```bash
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

### 4. Open the UI

Navigate to `http://127.0.0.1:8000` in your browser.

### 5. Run the competition demo

```bash
python demo.py
```

---

## Project Structure

```
nexus/
├── backend/                  # FastAPI server + all AI subsystems
│   ├── agent/                # Autonomous DAG planner & orchestrator
│   ├── demo/                 # Competition demo workflow
│   ├── ingestion/            # Document validation, extraction, chunking
│   ├── interfaces/           # OCR, LLM, vision, ASR adapters
│   ├── knowledge/            # Vector index & retrieval
│   ├── models_local/         # On-device model loading (ORT + GenAI)
│   ├── network/              # Local-only network guard
│   ├── optimization/         # NPU detection & provider selection
│   ├── tools/                # Agent tool registry (allowlisted only)
│   ├── verification/         # Grounded evidence & claim verification
│   └── main.py               # Application entry point
├── benchmarks/               # Reproducible benchmark suite
│   ├── suite.py              # Run: python benchmarks/suite.py
│   ├── generate_charts.py    # Run: python benchmarks/generate_charts.py
│   ├── raw/                  # Raw timing JSON files
│   ├── results/              # Summary JSON + CSV
│   └── charts/               # Generated PNG charts
├── demo_data/                # Synthetic industrial inspection dataset
├── docs/                     # All project documentation
│   ├── ARCHITECTURE.md
│   ├── MODEL_SELECTION.md
│   ├── SNAPDRAGON_OPTIMIZATION.md
│   ├── BENCHMARKS.md
│   ├── SECURITY.md
│   ├── DEMO_GUIDE.md
│   ├── LIMITATIONS.md
│   ├── FINAL_READINESS_REPORT.md
│   └── guides/               # Step-by-step setup guides
├── frontend/                 # HTML/CSS/JS single-page application
├── scripts/                  # Utility scripts
│   └── generate_demo_data.py # Regenerate synthetic demo dataset
├── tests/                    # 133+ automated tests
├── data/                     # Runtime data (documents, reports, db)
├── demo.py                   # CLI competition demo runner
└── run.py                    # Production server launcher
```

---

## Running the Benchmark Suite

```bash
# Run all component benchmarks (5 iterations each)
python benchmarks/suite.py --iterations 5

# Regenerate performance charts
python benchmarks/generate_charts.py
```

Results are written to `benchmarks/results/` and `benchmarks/charts/`.  
See [`docs/BENCHMARKS.md`](docs/BENCHMARKS.md) for full methodology and current results.

---

## Running Tests

```bash
pytest tests/ -v
```

The test suite covers 133+ cases across document ingestion, agent planning, security vectors, demo workflow, and API routes.

---

## Documentation Index

| Document | Purpose |
|---|---|
| [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) | System architecture & data flow |
| [`docs/MODEL_SELECTION.md`](docs/MODEL_SELECTION.md) | Model choices, hardware compatibility |
| [`docs/SNAPDRAGON_OPTIMIZATION.md`](docs/SNAPDRAGON_OPTIMIZATION.md) | NPU acceleration strategy |
| [`docs/BENCHMARKS.md`](docs/BENCHMARKS.md) | Empirical performance measurements |
| [`docs/SECURITY.md`](docs/SECURITY.md) | Threat model & security review |
| [`docs/DEMO_GUIDE.md`](docs/DEMO_GUIDE.md) | Competition demo walkthrough |
| [`docs/LIMITATIONS.md`](docs/LIMITATIONS.md) | Known limitations & future work |
| [`docs/FINAL_READINESS_REPORT.md`](docs/FINAL_READINESS_REPORT.md) | Final technical readiness checklist |
| [`docs/guides/INSTALLATION.md`](docs/guides/INSTALLATION.md) | General installation guide |
| [`docs/guides/WINDOWS_SETUP.md`](docs/guides/WINDOWS_SETUP.md) | Windows 11 configuration |
| [`docs/guides/SNAPDRAGON_SETUP.md`](docs/guides/SNAPDRAGON_SETUP.md) | Snapdragon NPU setup |
| [`docs/guides/MODEL_SETUP.md`](docs/guides/MODEL_SETUP.md) | Downloading & configuring AI models |
| [`docs/guides/OFFLINE_MODE.md`](docs/guides/OFFLINE_MODE.md) | Air-gap configuration |
| [`docs/guides/DEMO_GUIDE.md`](docs/guides/DEMO_GUIDE.md) | Running the competition demo |
| [`docs/guides/TROUBLESHOOTING.md`](docs/guides/TROUBLESHOOTING.md) | Common issues & solutions |

---

## Privacy & Air-Gap Guarantee

NEXUS enforces air-gap at the Python socket layer:

```
LocalNetworkGuard intercepts ALL outbound connections.
Remote addresses → StrictLocalOnlyViolationError (blocked + logged).
Only 127.0.0.1 / localhost is permitted.
```

No data ever leaves the device. No model weights are downloaded at runtime. No API keys required.

---

## License

MIT License — see [LICENSE](LICENSE).
