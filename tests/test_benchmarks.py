"""
NEXUS Phase 10 Tests: Snapdragon Optimization & Evidence-Driven Benchmark Engine
Validates:
- Hardware and execution provider auditing
- Multi-iteration empirical latency calculation (min, max, mean, median, p95)
- Honest hardware reporting (zero fabricated NPU claims on x86_64 hosts)
- Disk persistence in data/benchmarks/benchmark_results.json
- REST API endpoints: GET /benchmarks/status, GET /benchmarks/latest, POST /benchmarks/run
"""

import json
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

from backend.main import create_app
from backend.optimization.benchmarks import (
    BenchmarkRunner,
    BENCHMARKS_FILE,
    audit_hardware_and_providers,
    compute_statistics,
)


@pytest.fixture
def test_client():
    app = create_app()
    with TestClient(app) as client:
        yield client


# =====================================================================
# 1. Hardware & Execution Provider Auditing Tests
# =====================================================================

def test_audit_hardware_and_providers():
    """Verifies that the hardware auditor inspects silicon architecture and ONNX execution providers."""
    hw = audit_hardware_and_providers()
    assert "host_architecture" in hw
    assert "operating_system" in hw
    assert "processor_name" in hw
    assert hw["physical_cpu_cores"] >= 1
    assert hw["logical_cpu_cores"] >= 1
    assert hw["total_ram_gb"] > 0
    assert hw["available_ram_gb"] > 0
    assert isinstance(hw["onnx_available_providers"], list)
    assert "CPUExecutionProvider" in hw["onnx_available_providers"]
    assert "Snapdragon X Elite" in hw["target_snapdragon_silicon"]
    assert hw["offline_mode"] is True


def test_honest_npu_reporting_on_host_architecture():
    """
    Verifies that the runtime does NOT fake NPU acceleration.
    On x86_64/AMD64 host, Hexagon NPU must be reported as absent.
    """
    hw = audit_hardware_and_providers()
    arch = hw["host_architecture"].lower()
    if arch not in ["arm64", "aarch64"]:
        assert hw["npu_present"] is False
        assert "absent" in hw["npu_status"].lower() or "n/a" in hw["npu_status"].lower()


# =====================================================================
# 2. Statistical Measurement Function Tests
# =====================================================================

def test_compute_statistics():
    """Verifies accurate calculation of min, max, mean, median, and 95th percentile."""
    # Test empty
    empty_stats = compute_statistics([])
    assert empty_stats["median_ms"] == 0.0

    # Test odd length
    latencies_odd = [10.0, 20.0, 30.0, 40.0, 50.0]
    stats_odd = compute_statistics(latencies_odd)
    assert stats_odd["min_ms"] == 10.0
    assert stats_odd["max_ms"] == 50.0
    assert stats_odd["mean_ms"] == 30.0
    assert stats_odd["median_ms"] == 30.0
    assert stats_odd["p95_ms"] == 50.0

    # Test even length
    latencies_even = [10.0, 20.0, 30.0, 40.0]
    stats_even = compute_statistics(latencies_even)
    assert stats_even["median_ms"] == 25.0


# =====================================================================
# 3. Model Benchmark Execution Tests
# =====================================================================

def test_benchmark_embeddings():
    """Verifies that all-MiniLM-L6-v2 embedding model benchmark runs and computes empirical metrics."""
    runner = BenchmarkRunner(iterations=2)
    res = runner.benchmark_embeddings()
    assert res["model_name"] == "all-MiniLM-L6-v2"
    assert res["iterations"] == 2
    assert res["cold_start_ms"] >= 0.0
    assert res["latency_stats"]["median_ms"] > 0.0
    assert res["rss_memory_mb"] > 0.0
    assert "CPUExecutionProvider" in res["execution_unit"]


def test_benchmark_speech_model():
    """Verifies that Whisper-Small-Quantized ASR benchmark runs with empirical measurements."""
    runner = BenchmarkRunner(iterations=2)
    res = runner.benchmark_speech_model()
    assert res["model_name"] == "Whisper-Small-Quantized"
    assert res["iterations"] == 2
    assert res["latency_stats"]["median_ms"] >= 0.0
    assert res["rss_memory_mb"] > 0.0


def test_benchmark_vision_model():
    """Verifies that OpenAI-CLIP vision benchmark runs with empirical measurements."""
    runner = BenchmarkRunner(iterations=2)
    res = runner.benchmark_vision_model()
    assert res["model_name"] == "OpenAI-CLIP-ViT-B32-Quantized"
    assert res["iterations"] == 2
    assert res["latency_stats"]["median_ms"] > 0.0
    assert res["rss_memory_mb"] > 0.0


def test_benchmark_ocr_model_truthful_unimplemented_status():
    """Verifies that EasyOCR is reported truthfully as awaiting offline HTP compilation."""
    runner = BenchmarkRunner(iterations=1)
    res = runner.benchmark_ocr_model()
    assert res["model_name"] == "EasyOCR (CRAFT + CRNN)"
    assert res["iterations"] == 0
    assert "HTP context binary pending" in res["execution_unit"]


def test_benchmark_run_all_and_persistence():
    """Verifies that running full benchmark suite persists valid JSON to BENCHMARKS_FILE."""
    runner = BenchmarkRunner(iterations=1)
    results = runner.run_all_benchmarks()
    assert "timestamp" in results
    assert "hardware_profile" in results
    assert len(results["models"]) == 5
    assert BENCHMARKS_FILE.exists()

    with open(BENCHMARKS_FILE, "r", encoding="utf-8") as f:
        saved = json.load(f)
    assert saved["timestamp"] == results["timestamp"]
    assert len(saved["models"]) == 5


# =====================================================================
# 4. REST API Endpoint Tests
# =====================================================================

def test_api_benchmarks_status(test_client):
    """Verifies GET /benchmarks/status returns hardware profile and providers."""
    response = test_client.get("/benchmarks/status")
    assert response.status_code == 200
    data = response.json()
    assert "host_architecture" in data
    assert "onnx_available_providers" in data
    assert "CPUExecutionProvider" in data["onnx_available_providers"]


def test_api_benchmarks_latest(test_client):
    """Verifies GET /benchmarks/latest retrieves persisted benchmark run."""
    response = test_client.get("/benchmarks/latest")
    assert response.status_code == 200
    data = response.json()
    assert "timestamp" in data
    assert "models" in data
    assert len(data["models"]) == 5


def test_api_benchmarks_run(test_client):
    """Verifies POST /benchmarks/run triggers a fresh benchmark run."""
    response = test_client.post("/benchmarks/run", json={"iterations": 1})
    assert response.status_code == 200
    data = response.json()
    assert data["iterations_per_model"] == 1
    assert len(data["models"]) == 5


def test_api_benchmarks_run_invalid_iterations_rejected(test_client):
    """Verifies POST /benchmarks/run rejects excessive iteration counts."""
    response = test_client.post("/benchmarks/run", json={"iterations": 50})
    assert response.status_code == 422
