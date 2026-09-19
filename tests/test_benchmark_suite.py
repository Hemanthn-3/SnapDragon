"""
NEXUS Phase 13: Benchmark Suite Automated Verification Tests
Verifies benchmark execution, measurement capture, JSON/CSV exports, and chart generation.
"""

import json
from pathlib import Path
import pytest
from benchmarks.suite import NexusBenchmarkSuite
from benchmarks.generate_charts import generate_all_charts, RESULTS_FILE, CHARTS_DIR


class TestBenchmarkSuite:
    """Automated tests for reproducible benchmark suite."""

    def test_benchmark_suite_individual_workloads(self):
        """Verify individual workload benchmarking functions return valid schema."""
        suite = NexusBenchmarkSuite(iterations=1)

        # 1. Test ASR benchmark
        asr_res = suite.benchmark_asr()
        assert asr_res["component"] == "ASR"
        assert asr_res["cold_latency_ms"] >= 0
        assert asr_res["median_latency_ms"] >= 0
        assert asr_res["memory_rss_mb"] > 0
        assert asr_res["status"] == "VERIFIED"

        # 2. Test Embedding benchmark
        emb_res = suite.benchmark_embedding()
        assert emb_res["component"] == "Embedding"
        assert emb_res["cold_latency_ms"] >= 0
        assert emb_res["median_latency_ms"] >= 0
        assert emb_res["status"] == "VERIFIED"

        # 3. Test Retrieval benchmark
        ret_res = suite.benchmark_retrieval()
        assert ret_res["component"] == "Retrieval"
        assert ret_res["cold_latency_ms"] >= 0
        assert ret_res["median_latency_ms"] >= 0
        assert ret_res["status"] == "VERIFIED"

        # 4. Test Vision benchmark
        vis_res = suite.benchmark_vision()
        assert vis_res["component"] == "Vision"
        assert vis_res["cold_latency_ms"] >= 0
        assert vis_res["median_latency_ms"] >= 0
        assert vis_res["status"] == "VERIFIED"

        # 5. Test OCR audit status
        ocr_res = suite.benchmark_ocr()
        assert ocr_res["component"] == "OCR"
        assert ocr_res["status"] == "NOT_YET_IMPLEMENTED"
        assert ocr_res["cold_latency_ms"] is None

    def test_benchmark_summary_files_exist_and_valid(self):
        """Verify benchmark summary JSON and CSV files exist and contain all 7 components."""
        assert RESULTS_FILE.exists(), f"{RESULTS_FILE} does not exist"

        with open(RESULTS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)

        assert "benchmark_suite_version" in data
        assert "hardware" in data
        assert "results" in data

        components = [r["component"] for r in data["results"]]
        expected = ["ASR", "OCR", "Embedding", "Retrieval", "LLM", "Vision", "Complete Agent Workflow"]
        for comp in expected:
            assert comp in components, f"Component {comp} missing from results"

        # Check CSV
        csv_file = RESULTS_FILE.parent / "benchmark_summary.csv"
        assert csv_file.exists()
        csv_text = csv_file.read_text(encoding="utf-8")
        assert "Component,Model,Runtime" in csv_text
        for comp in expected:
            assert comp in csv_text

    def test_chart_generation(self):
        """Verify chart generation script produces all four expected PNG charts."""
        generate_all_charts()

        expected_charts = [
            "latency_comparison.png",
            "memory_rss_profile.png",
            "workflow_breakdown.png",
            "cold_vs_warm.png",
        ]

        for chart_name in expected_charts:
            chart_path = CHARTS_DIR / chart_name
            assert chart_path.exists(), f"Chart {chart_name} was not generated"
            # Ensure file is a non-empty image
            assert chart_path.stat().st_size > 1000, f"Chart {chart_name} is too small or corrupt"
