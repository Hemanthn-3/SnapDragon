"""
NEXUS Phase 15: Deterministic Competition Demonstration Runner
Executes the end-to-end multimodal workflow over synthetic demonstration data.
Visibly presents all 10 cognitive milestones, air-gap hardware telemetry,
and opens the generated action report.

Usage:
    python demo.py
"""

import os
import sys
import time
from pathlib import Path

# Ensure project root is in sys.path
BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

from backend.demo.service import demo_orchestrator, DEMO_GOAL
from backend.database import init_database, SessionLocal


SEPARATOR = "=" * 80
SUB_SEPARATOR = "-" * 80


def print_banner():
    print(SEPARATOR)
    print("  NEXUS: Offline Multimodal Work Agent — Competition Demonstration")
    print("  Target Silicon: Qualcomm Snapdragon X Elite (Windows 11 on ARM64)")
    print("  Execution Mode: 100% AIR-GAPPED LOCAL ONLY")
    print(SEPARATOR)
    print()


def print_stage_header(step: int, name: str):
    print(f"\n[STAGE {step}/10] >>> {name.upper()} <<<")
    print(SUB_SEPARATOR)


def run_demo():
    print_banner()

    # Initialize local database
    init_database()

    print("[DEMO_MODE] Initializing reproducible synthetic dataset...")
    with SessionLocal() as db:
        load_res = demo_orchestrator.load_demo_dataset(db=db)
    
    print(f"Loaded {load_res['total_documents']} demonstration assets into local repository:")
    for doc in load_res["loaded_documents"]:
        print(f"  * {doc['filename']:<35} ({doc['file_type'].upper()}, {doc['chunks']} chunks, {doc['size_bytes']} bytes)")
    print()

    # Execute 10-stage demo
    print(SEPARATOR)
    print("STARTING DETERMINISTIC WORKFLOW EXECUTION")
    print(SEPARATOR)

    with SessionLocal() as db:
        res = demo_orchestrator.run_competition_demo(db=db)

    stages = res["stages"]

    # 1. User Goal
    s1 = stages[0]
    print_stage_header(1, s1["name"])
    print(f"USER: \"{s1['details']['goal']}\"")
    print(f"Target: {s1['details']['target_equipment']}")

    # 2. Agent Plan
    s2 = stages[1]
    print_stage_header(2, s2["name"])
    print(f"Decomposed DAG Plan ({s2['details']['total_tasks']} Tasks):")
    for t in s2["details"]["tasks"]:
        dep_str = f" <- [{', '.join(t['dependencies'])}]" if t['dependencies'] else " (Root)"
        print(f"  [{t['id']}] {t['type']:<20} : {t['description']}{dep_str}")

    # 3. Document Discovery
    s3 = stages[2]
    print_stage_header(3, s3["name"])
    print(f"Discovered {s3['details']['package_count']} inspection package files:")
    for f in s3['details']['files']:
        print(f"  + {f}")

    # 4. OCR
    s4 = stages[3]
    print_stage_header(4, s4["name"])
    print(f"Target Document: {s4['details']['target_document']}")
    print(f"Engine:          {s4['details']['ocr_engine']}")
    print(f"Extracted Text:  \"{s4['details']['extracted_key_text']}\"")
    print(f"Detected Value:  {s4['details']['detected_parameter']} = {s4['details']['detected_value']}")
    print(f"Signoff:         {s4['details']['technician_signoff']}")

    # 5. Retrieval
    s5 = stages[4]
    print_stage_header(5, s5["name"])
    print(f"Executed {s5['details']['queries_executed']} semantic searches across vector store:")
    for rf in s5['details']['top_evidence_retrieved']:
        print(f"  Query: \"{rf['query']}\"")
        print(f"    -> Source: {rf['document']} (sim: {rf['similarity']:.3f})")
        print(f"    -> Excerpt: {rf['excerpt'].strip()}...")

    # 6. Vision
    s6 = stages[5]
    print_stage_header(6, s6["name"])
    print(f"Target Image:  {s6['details']['target_image']}")
    print(f"Model:         {s6['details']['model']}")
    print(f"OBSERVED:      Resolution {s6['details']['observed']['resolution']} | Format {s6['details']['observed']['format']}")
    print(f"               Visual Tags: {', '.join(s6['details']['observed']['top_visual_tags'])}")
    print(f"INFERRED:      {s6['details']['inferred']['visual_finding']}")
    print(f"               {s6['details']['inferred']['contradiction_noted']}")

    # 7. Local Reasoning
    s7 = stages[6]
    print_stage_header(7, s7["name"])
    print(f"Reasoning Engine: {s7['details']['model']}")
    print(f"Identified {s7['details']['total_issues_identified']} engineering discrepancies:")
    for d in s7['details']['discrepancies']:
        print(f"  [{d['issue_id']}] {d['severity']:<8} | {d['parameter']}")
        print(f"       Baseline: {d['reference_limit']}")
        print(f"       Measured: {d['measured_value']} ({d['delta']})")
        print(f"       Hazard:   {d['hazard']}")

    # 8. Evidence
    s8 = stages[7]
    print_stage_header(8, s8["name"])
    print(f"Compiled {s8['details']['evidence_chain_count']} traceable evidence records:")
    for ev in s8['details']['evidence_items']:
        print(f"  [{ev['finding_id']}] Claim: \"{ev['claim']}\"")
        print(f"       Source:    {ev['source_doc']} -> \"{ev['citation']}\"")
        print(f"       Benchmark: {ev['grounding_doc']} -> \"{ev['reference_citation']}\"")

    # 9. Verification
    s9 = stages[8]
    print_stage_header(9, s9["name"])
    print(f"Verified {s9['details']['claims_verified']} critical claims:")
    for v in s9['details']['results']:
        print(f"  * Claim: \"{v['claim']}\"")
        print(f"    Status: {v['verification_status']} (Confidence: {v['confidence']:.2f})")
        print(f"    Note:   {v['explanation']}")

    # 10. Report Generation
    s10 = stages[9]
    print_stage_header(10, s10["name"])
    print(f"Title:         {s10['details']['title']}")
    print(f"Export Path:   {s10['details']['exported_path']}")
    print(f"Size:          {s10['details']['bytes_written']} bytes written to disk")
    print(f"Report ID:     {s10['details']['report_id']}")

    # Air-Gap Hardware Telemetry Display
    hw = res["hardware_status"]
    print()
    print(SEPARATOR)
    print("  AIR-GAPPED HARDWARE EXECUTION STATUS")
    print(SEPARATOR)
    print(f"  NETWORK:  {hw['network']} ({hw['internet_detail']})")
    print(f"  AI:       {hw['ai']} (Cloud AI: {hw['cloud_ai']})")
    print(f"  NPU:      {hw['npu']}")
    print(f"  RUNTIME:  {hw['runtime_providers']}")
    print(f"  TOTAL ELAPSED: {res['duration_ms']:.2f} ms")
    print(SEPARATOR)
    print()

    # Open Generated Report
    report_file = Path(s10["details"]["exported_path"])
    print(f"[REPORT OPENER] Opening generated report: {report_file.resolve()}")
    print(SUB_SEPARATOR)
    if report_file.exists():
        content = report_file.read_text(encoding="utf-8")
        print(content)
    else:
        print(s10["details"]["content"])
    print(SUB_SEPARATOR)
    print(f"Report successfully generated and saved to: {report_file.resolve()}")


if __name__ == "__main__":
    run_demo()
