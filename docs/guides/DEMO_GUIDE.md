# NEXUS Competition Demo — Step-by-Step Guide

**For Competition Judges and Evaluators**

---

## What You Will See

The NEXUS competition demo executes a complete multimodal industrial inspection workflow:

1. **10 clearly labeled cognitive stages** execute in sequence
2. **Real document processing** — 5 synthetic files are analyzed
3. **5 engineering discrepancies** are identified and cross-verified
4. **Hardware status** is displayed from actual runtime state
5. **An executive action report** is generated and displayed

The entire workflow runs **offline** — you can disconnect network during the demo to confirm.

---

## Before You Start

Ensure the backend is running:

```bash
cd nexus/
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

You should see:
```
INFO: Starting NEXUS v1.0.0 [production]
INFO: Offline Mode: True
INFO: Application startup complete.
```

---

## Option 1: CLI Demo (Best for Judges)

```bash
python demo.py
```

### What You Will See

```
╔══════════════════════════════════════════════════════╗
║         NEXUS COMPETITION DEMO v1.0                  ║
║     Offline Multimodal Industrial Inspection         ║
╚══════════════════════════════════════════════════════╝

Goal: Analyze the inspection package, compare it against the
      reference manual, identify issues, verify the findings,
      and create an action report.

[1/10] ✓ User Goal         — Goal submitted
[2/10] ✓ Agent Plan        — 6-task DAG generated (Kahn validated)
[3/10] ✓ Document Discovery — 5 documents found in corpus
[4/10] ✓ OCR              — "Fastener Bolt Torque Applied: 95 Nm" extracted
[5/10] ✓ Retrieval         — 245 Bar exceeds 220 Bar safety limit (k=3)
[6/10] ✓ Vision            — surface_pitting, micro_abrasion detected
[7/10] ✓ Local Reasoning   — 5 discrepancies synthesized
[8/10] ✓ Evidence          — Citations linked to source chunks
[9/10] ✓ Verification      — All 5 findings grounded and confirmed
[10/10] ✓ Report           — Report written to data/reports/

╔════════════════════════════════════╗
║      NEXUS HARDWARE STATUS         ║
╠════════════════════════════════════╣
║  NETWORK    ●  OFFLINE             ║
║  AI         ●  LOCAL               ║
║  NPU        ●  ACTIVE / STANDBY    ║
╚════════════════════════════════════╝

══ GENERATED REPORT ══════════════════════════════════════

# Turbine TR-900 — Action Report
...
```

---

## Option 2: Web UI Demo

1. Start the backend (see above)
2. Open `http://127.0.0.1:8000`
3. Find **⚡ Competition Demo Mode** card on the home screen
4. Click **Run Competition Demo**
5. Watch the stage tracker advance through all 10 stages
6. The hardware status panel and report appear when complete

---

## What Each Stage Proves

| Stage | Cognitive Capability Demonstrated |
|---|---|
| 1 — User Goal | Natural language goal intake (voice or text) |
| 2 — Agent Plan | Autonomous multi-step planning with DAG validation |
| 3 — Document Discovery | Multi-format document corpus management |
| 4 — OCR | Image-based text extraction (EasyOCR / Whisper pipeline) |
| 5 — Retrieval | Semantic search over indexed document embeddings |
| 6 — Vision | Zero-shot visual analysis via CLIP ViT-B/32 |
| 7 — Local Reasoning | On-device LLM synthesis across multiple evidence sources |
| 8 — Evidence | Grounded citation building (chunk IDs, page refs, scores) |
| 9 — Verification | Independent cross-verification of each claim vs. evidence |
| 10 — Report | Structured executive report generation |

---

## The 5 Discrepancies NEXUS Must Find

| # | Discrepancy | Source | How NEXUS Finds It |
|---|---|---|---|
| 1 | Hydraulic pressure: 245 Bar (limit: 220 Bar) | inspection_report.pdf | Retrieval + reasoning |
| 2 | Wrong lubricant: Mineral SAE 30 (required: ISO VG 46) | inspection_report.pdf | Retrieval + reasoning |
| 3 | Bolt torque: 95 Nm (required: 125 Nm ± 5 Nm) | scanned_maintenance_log.png | OCR |
| 4 | Housing surface: pitting observed (checklist: "Pristine") | bearing_assembly_inspection.png | Vision (CLIP) |
| 5 | Vibration: 5.8 mm/s (alarm: > 4.2 mm/s) | sensor_telemetry.json | Retrieval + reasoning |

---

## Verifying Authenticity

**No progress bars are timers.** Each stage marker advances only when the actual operation completes.

**To confirm offline operation**: Disconnect your network cable or disable WiFi before running the demo. Everything continues to work.

**To confirm local AI**: Check the NPU/CPU utilization in Windows Task Manager — you will see Python process CPU activity as the LLM generates tokens.

**To see raw results**: 
```bash
cat data/reports/turbine_action_report.md
```

**To see evidence citations**:
```bash
curl http://127.0.0.1:8000/api/demo/status
```

---

## Resetting the Demo

To run again from a clean state:

```bash
# Delete the demo run record (keeps ingested documents)
curl -X DELETE http://127.0.0.1:8000/api/demo/reset

# Or regenerate all demo data:
python scripts/generate_demo_data.py
```

---

## Common Questions

**Q: Is the report pre-written?**  
A: No. The report is generated fresh on every run by the local LLM reasoning over the retrieved document evidence.

**Q: Are the discrepancies hardcoded into the output?**  
A: No. They are embedded in the synthetic documents. NEXUS must actually read, OCR, and retrieve them to surface them.

**Q: What if the NPU badge shows STANDBY instead of ACTIVE?**  
A: STANDBY means the host is not Snapdragon ARM64 hardware. All processing still runs locally on CPU. The demo outputs are identical — only the hardware acceleration tier differs.

**Q: Can I upload my own documents?**  
A: Yes — use the Documents screen to upload PDFs, DOCX, TXT, or images, then create a custom task on the New Task screen.
