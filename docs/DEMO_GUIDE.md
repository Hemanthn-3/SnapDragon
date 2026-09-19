# NEXUS Demo Guide — Competition Walkthrough

**Phase 15: Deterministic Industrial Inspection Demo**  
**Data**: 100% Synthetic · Reproducible · No Fabricated Metrics

---

## Overview

The NEXUS competition demo executes a complete multimodal industrial inspection workflow against synthetic data representing a Turbine Rotor Assembly inspection package. The demo visibly walks through all **10 cognitive stages** of the NEXUS agent pipeline and concludes by displaying hardware privacy status and the generated action report.

The synthetic dataset was designed with **5 deliberate, detectable engineering discrepancies** that NEXUS must identify, cross-reference, and report — requiring real document retrieval, OCR, vision analysis, local reasoning, and grounded verification.

---

## Synthetic Demo Dataset

Location: `demo_data/`

| File | Type | Contents |
|---|---|---|
| `reference_manual.txt` | Text | TR-900 turbine technical specifications (nominal values, alarm thresholds) |
| `inspection_report.pdf` | PDF | Field inspection log for Unit TR-900 (hydraulic pressure + lubricant readings) |
| `scanned_maintenance_log.png` | PNG | Scanned service log table (bolt torque applied value) |
| `bearing_assembly_inspection.png` | PNG | Bearing housing diagram (surface condition photograph) |
| `sensor_telemetry.json` | JSON | SCADA sensor log (vibration readings per channel) |

### Deliberate Discrepancies

| # | Finding | Reference Spec | Measured/Observed | Source |
|---|---|---|---|---|
| 1 | Hydraulic pressure exceeds safety limit | ≤ 220 Bar max | **245 Bar** recorded | `inspection_report.pdf` |
| 2 | Wrong lubricant type applied | ISO VG 46 Synthetic | **Mineral SAE 30** used | `inspection_report.pdf` |
| 3 | Fastener torque below specification | 125 Nm ± 5 Nm | **95 Nm** applied | `scanned_maintenance_log.png` (OCR) |
| 4 | Bearing housing surface defects | "Pristine" (checklist) | **Pitting and micro-abrasion observed** | `bearing_assembly_inspection.png` (Vision) |
| 5 | Vibration exceeds alarm threshold | ≤ 4.2 mm/s | **5.8 mm/s** (VIB_CH02) | `sensor_telemetry.json` |

---

## Running the Demo

### Option A: CLI Runner (Recommended for Competition)

```bash
cd nexus/
python demo.py
```

The CLI runner:
1. Generates or verifies the synthetic demo dataset
2. Loads and ingests all 5 demo documents into NEXUS
3. Executes the 10-stage cognitive workflow
4. Displays real-time stage progress with status badges
5. Prints the final hardware status block (NETWORK/AI/NPU)
6. Displays the full generated action report

Expected runtime: 15–120 seconds depending on hardware (NPU vs CPU)

### Option B: Web UI Demo

1. Start the backend:
   ```bash
   python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
   ```
2. Open `http://127.0.0.1:8000` in your browser
3. On the Home screen, locate the **⚡ Competition Demo Mode** card
4. Click **Run Competition Demo**
5. Watch the 10-stage progress tracker animate in real time
6. The hardware status and generated report appear when complete

### Option C: API (for integration testing)

```bash
# Step 1: Setup demo data
curl -X POST http://127.0.0.1:8000/api/demo/setup

# Step 2: Run the full workflow
curl -X POST http://127.0.0.1:8000/api/demo/run

# Step 3: Get status
curl http://127.0.0.1:8000/api/demo/status

# Step 4: Get the generated report
curl http://127.0.0.1:8000/api/demo/report
```

---

## The 10 Cognitive Stages

| Stage | Name | What Happens |
|---|---|---|
| 1 | **User Goal** | Goal text submitted: *"Analyze the inspection package, compare against reference manual, identify issues, verify findings, create action report."* |
| 2 | **Agent Plan** | Llama-3.2-1B generates a structured DAG plan with tasks and dependencies |
| 3 | **Document Discovery** | Agent enumerates the 5 ingested demo documents |
| 4 | **OCR** | EasyOCR reads the scanned maintenance log image, extracts "95 Nm" bolt torque |
| 5 | **Retrieval** | Semantic search against reference manual finds pressure (220 Bar limit) and vibration (4.2 mm/s limit) thresholds |
| 6 | **Vision** | CLIP ViT-B/32 analyzes bearing assembly image, tags: `surface_pitting`, `micro_abrasion` |
| 7 | **Local Reasoning** | Llama-3.2-1B synthesizes OCR, retrieval, and vision outputs into a structured findings table |
| 8 | **Evidence** | All citations are linked: chunk IDs, document names, page numbers, similarity scores |
| 9 | **Verification** | Each of the 5 discrepancies is cross-verified against the reference manual context |
| 10 | **Report Generation** | Executive action report generated to `data/reports/turbine_action_report.md` |

---

## Final Hardware Status Display

After stage 10, NEXUS displays actual hardware status:

```
╔════════════════════════════════════╗
║        NEXUS HARDWARE STATUS       ║
╠════════════════════════════════════╣
║  NETWORK    ●  OFFLINE             ║
║  AI         ●  LOCAL               ║
║  NPU        ●  [ACTIVE / STANDBY]  ║
╚════════════════════════════════════╝
```

- **NETWORK OFFLINE**: `LocalNetworkGuard` confirmed zero egress attempts
- **AI LOCAL**: All inference executed on-device (no cloud API calls)
- **NPU [ACTIVE]**: Displayed when `QNNExecutionProvider` is detected (Snapdragon hardware)
- **NPU [STANDBY]**: Displayed on x64 development hosts (CPU fallback active)

These values are **read from actual runtime state** — never hardcoded.

---

## Regenerating the Demo Dataset

If the `demo_data/` directory is missing or corrupted:

```bash
python scripts/generate_demo_data.py
```

This recreates all 5 synthetic files deterministically. The script generates the PNG images with embedded text, the inspection report PDF, the structured telemetry JSON, and the reference manual text — all with the embedded discrepancies described above.

---

## Automated Demo Tests

```bash
pytest tests/test_demo.py -v
```

Tests verify:
- `demo_data/` files exist and match schemas
- All 5 files ingest successfully
- The 10-stage workflow completes
- The generated report references all 5 discrepancies
- Hardware status confirms `OFFLINE` and `LOCAL`

---

## What Judges Should Look For

1. **No Fake Progress**: The progress bar reflects actual pipeline stage completion — not a timer.
2. **Real Document Analysis**: Each finding traces to a real chunk from a real ingested file.
3. **Offline Operation**: Watch the network monitor — zero external requests during demo.
4. **NPU Status**: On Snapdragon hardware, the NPU badge turns green showing actual hardware acceleration.
5. **Grounded Report**: The final report cites chunk IDs, page numbers, and similarity scores — not hallucinated summaries.
