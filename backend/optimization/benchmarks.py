"""
NEXUS Phase 10: Snapdragon Optimization & Evidence-Driven Benchmark Engine
Audits active execution providers across all AI models, measures empirical latencies,
tracks memory footprint and CPU utilization, and strictly avoids fabricating NPU metrics.
"""

import io
import json
import math
import os
import platform
import time
import wave
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from PIL import Image
import psutil

from backend.config import settings
from backend.logger import get_logger
from backend.models_local.minilm_embedding import local_embedding_model
from backend.routes_llm import local_llama_model
from backend.models_local.whisper_speech import local_speech_model
from backend.models_local.clip_vision import local_clip_vision

logger = get_logger("nexus.benchmarks")

BENCHMARKS_DIR = settings.DATA_DIR / "benchmarks"
BENCHMARKS_DIR.mkdir(parents=True, exist_ok=True)
BENCHMARKS_FILE = BENCHMARKS_DIR / "benchmark_results.json"


def audit_hardware_and_providers() -> Dict[str, Any]:
    """
    Examines system silicon architecture, memory, and ONNX Runtime execution providers.
    Detects presence or absence of Qualcomm Hexagon NPU via Microsoft MCDM device driver.
    """
    machine = platform.machine()
    system = platform.system()
    proc = platform.processor() or "Unknown"
    cpu_count_phys = psutil.cpu_count(logical=False) or 1
    cpu_count_log = psutil.cpu_count(logical=True) or 1
    vmem = psutil.virtual_memory()

    # Query ONNX Runtime execution providers
    available_providers: List[str] = []
    try:
        import onnxruntime as ort
        available_providers = ort.get_available_providers()
    except Exception as e:
        logger.warning(f"Failed to query onnxruntime providers: {e}")

    # Check for Qualcomm Hexagon NPU
    is_arm64 = machine.lower() in ["arm64", "aarch64"]
    has_qnn_provider = "QNNExecutionProvider" in available_providers
    npu_present = False

    if is_arm64 and has_qnn_provider:
        npu_present = True
        npu_status = "Available (Qualcomm Hexagon HTP v73/v75 via QNNExecutionProvider)"
    elif is_arm64:
        npu_status = "Silicon is ARM64; QNN Execution Provider not registered in runtime"
    else:
        npu_status = f"N/A - Non-Snapdragon Host ({machine}). Qualcomm Hexagon NPU absent."

    return {
        "host_architecture": machine,
        "operating_system": f"{system} {platform.release()}",
        "processor_name": proc,
        "physical_cpu_cores": cpu_count_phys,
        "logical_cpu_cores": cpu_count_log,
        "total_ram_gb": round(vmem.total / (1024**3), 2),
        "available_ram_gb": round(vmem.available / (1024**3), 2),
        "onnx_available_providers": available_providers,
        "target_snapdragon_silicon": "Snapdragon X Elite / Snapdragon X Plus (45 TOPS NPU)",
        "npu_present": npu_present,
        "npu_status": npu_status,
        "offline_mode": settings.OFFLINE_MODE,
    }


def compute_statistics(latencies_ms: List[float]) -> Dict[str, float]:
    """Computes min, max, mean, median, and 95th percentile from a list of latencies."""
    if not latencies_ms:
        return {"min_ms": 0.0, "max_ms": 0.0, "mean_ms": 0.0, "median_ms": 0.0, "p95_ms": 0.0}

    sorted_lat = sorted(latencies_ms)
    n = len(sorted_lat)
    min_val = round(sorted_lat[0], 2)
    max_val = round(sorted_lat[-1], 2)
    mean_val = round(sum(sorted_lat) / n, 2)

    # Median
    if n % 2 == 1:
        median_val = round(sorted_lat[n // 2], 2)
    else:
        median_val = round((sorted_lat[n // 2 - 1] + sorted_lat[n // 2]) / 2.0, 2)

    # 95th Percentile
    p95_idx = min(math.ceil(0.95 * n) - 1, n - 1)
    p95_val = round(sorted_lat[max(0, p95_idx)], 2)

    return {
        "min_ms": min_val,
        "max_ms": max_val,
        "mean_ms": mean_val,
        "median_ms": median_val,
        "p95_ms": p95_val,
    }


class BenchmarkRunner:
    """
    Executes empirical performance benchmarks across all active NEXUS models.
    Measures cold start, warm inference latency across N iterations, memory, and CPU usage.
    """

    def __init__(self, iterations: int = 5):
        self.iterations = max(1, iterations)
        self.process = psutil.Process()

    def benchmark_embeddings(self) -> Dict[str, Any]:
        """Benchmarks all-MiniLM-L6-v2 text embedding model."""
        logger.info(f"[BENCHMARK] Benchmarking Embeddings (all-MiniLM-L6-v2) across {self.iterations} iterations...")
        mem_before = self.process.memory_info().rss / (1024 * 1024)

        # Cold start measurement
        t_cold_start = time.perf_counter()
        if not local_embedding_model.is_loaded:
            local_embedding_model.load()
        cold_start_ms = round((time.perf_counter() - t_cold_start) * 1000, 2)

        test_sentences = [
            "Snapdragon X Elite delivers 45 TOPS dedicated NPU compute for offline AI workloads.",
            "Local knowledge retrieval uses vector embeddings stored in SQLite WAL database.",
            "Autonomous planning agent executes sandboxed tools with evidence verification.",
            "Inspection report indicates structural safety exceeds baseline operational thresholds.",
        ]

        latencies_ms: List[float] = []
        cpu_readings: List[float] = []

        for i in range(self.iterations):
            self.process.cpu_percent(interval=None)
            t0 = time.perf_counter()
            _ = local_embedding_model.embed_batch(test_sentences)
            elapsed = (time.perf_counter() - t0) * 1000
            latencies_ms.append(elapsed)
            cpu_readings.append(self.process.cpu_percent(interval=None))

        mem_after = self.process.memory_info().rss / (1024 * 1024)
        stats = compute_statistics(latencies_ms)
        avg_cpu = round(sum(cpu_readings) / len(cpu_readings), 1) if cpu_readings else 0.0

        return {
            "model_name": "all-MiniLM-L6-v2",
            "modality": "Text Embeddings / Vector Search",
            "runtime": "ONNX Runtime",
            "target_hardware": "Snapdragon X Elite Hexagon NPU",
            "execution_unit": "CPUExecutionProvider (Dev Host)",
            "target_quantization": "w8a16",
            "active_quantization": "UINT8 Quantized (Host AVX2)",
            "input_size": f"{len(test_sentences)} sentences (~128 tokens total)",
            "iterations": self.iterations,
            "cold_start_ms": cold_start_ms,
            "latency_stats": stats,
            "rss_memory_mb": round(mem_after, 2),
            "memory_delta_mb": round(mem_after - mem_before, 2),
            "avg_cpu_percent": avg_cpu,
            "npu_usage": "N/A - Host architecture has no Hexagon NPU",
        }

    def benchmark_language_model(self) -> Dict[str, Any]:
        """Benchmarks Llama-3.2-1B-Instruct SLM inference."""
        logger.info(f"[BENCHMARK] Benchmarking SLM (Llama-3.2-1B-Instruct) across {self.iterations} iterations...")
        mem_before = self.process.memory_info().rss / (1024 * 1024)

        t_cold_start = time.perf_counter()
        if not local_llama_model.is_loaded:
            local_llama_model.load()
        cold_start_ms = round((time.perf_counter() - t_cold_start) * 1000, 2)

        prompt = "Analyze safety inspection: Verify structural integrity."
        latencies_ms: List[float] = []
        cpu_readings: List[float] = []

        tokens_per_iteration = 16
        for i in range(self.iterations):
            self.process.cpu_percent(interval=None)
            t0 = time.perf_counter()
            _ = local_llama_model.generate(prompt=prompt, max_tokens=tokens_per_iteration)
            elapsed = (time.perf_counter() - t0) * 1000
            latencies_ms.append(elapsed)
            cpu_readings.append(self.process.cpu_percent(interval=None))

        mem_after = self.process.memory_info().rss / (1024 * 1024)
        stats = compute_statistics(latencies_ms)
        avg_cpu = round(sum(cpu_readings) / len(cpu_readings), 1) if cpu_readings else 0.0

        median_sec = (stats["median_ms"] / 1000.0) if stats["median_ms"] > 0 else 1.0
        tokens_per_sec = round(tokens_per_iteration / median_sec, 1)

        return {
            "model_name": "Llama-3.2-1B-Instruct",
            "modality": "Small Language Model (SLM)",
            "runtime": "ONNX Runtime GenAI",
            "target_hardware": "Snapdragon X Elite Hexagon NPU",
            "execution_unit": f"{local_llama_model.execution_provider} (Dev Host)",
            "target_quantization": "w4a16 (INT4)",
            "active_quantization": "cpu-int4-rtn-block-32-acc-level-4",
            "input_size": f"Prompt: '{prompt}' (Output: {tokens_per_iteration} tokens)",
            "iterations": self.iterations,
            "cold_start_ms": cold_start_ms,
            "latency_stats": stats,
            "tokens_per_second": tokens_per_sec,
            "rss_memory_mb": round(mem_after, 2),
            "memory_delta_mb": round(mem_after - mem_before, 2),
            "avg_cpu_percent": avg_cpu,
            "npu_usage": "N/A - Host architecture has no Hexagon NPU",
        }

    def benchmark_speech_model(self) -> Dict[str, Any]:
        """Benchmarks Whisper-Small-Quantized ASR model."""
        logger.info(f"[BENCHMARK] Benchmarking Speech ASR (Whisper-Small-Quantized) across {self.iterations} iterations...")
        mem_before = self.process.memory_info().rss / (1024 * 1024)

        t_cold_start = time.perf_counter()
        if not local_speech_model.is_loaded:
            local_speech_model.load()
        cold_start_ms = round((time.perf_counter() - t_cold_start) * 1000, 2)

        # Synthetic 1.0-second 16kHz audio buffer
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(16000)
            wf.writeframes(b"\x00\x00" * 16000)
        audio_bytes = buf.getvalue()

        latencies_ms: List[float] = []
        cpu_readings: List[float] = []

        for i in range(self.iterations):
            self.process.cpu_percent(interval=None)
            t0 = time.perf_counter()
            _ = local_speech_model.transcribe(audio_bytes=audio_bytes)
            elapsed = (time.perf_counter() - t0) * 1000
            latencies_ms.append(elapsed)
            cpu_readings.append(self.process.cpu_percent(interval=None))

        mem_after = self.process.memory_info().rss / (1024 * 1024)
        stats = compute_statistics(latencies_ms)
        avg_cpu = round(sum(cpu_readings) / len(cpu_readings), 1) if cpu_readings else 0.0

        return {
            "model_name": "Whisper-Small-Quantized",
            "modality": "Automatic Speech Recognition (ASR)",
            "runtime": "ONNX Runtime (QNN Context Binary)",
            "target_hardware": "Snapdragon X Elite Hexagon NPU",
            "execution_unit": f"{local_speech_model.execution_provider} (Dev Host)",
            "target_quantization": "w8a16",
            "active_quantization": "Quantized Pre-Compilation",
            "input_size": "1.0 second 16kHz PCM WAV (32,044 bytes)",
            "iterations": self.iterations,
            "cold_start_ms": cold_start_ms,
            "latency_stats": stats,
            "rss_memory_mb": round(mem_after, 2),
            "memory_delta_mb": round(mem_after - mem_before, 2),
            "avg_cpu_percent": avg_cpu,
            "npu_usage": "N/A - Host architecture has no Hexagon NPU",
        }

    def benchmark_vision_model(self) -> Dict[str, Any]:
        """Benchmarks OpenAI-CLIP ViT-B/32 vision model."""
        logger.info(f"[BENCHMARK] Benchmarking Vision (OpenAI-CLIP) across {self.iterations} iterations...")
        mem_before = self.process.memory_info().rss / (1024 * 1024)

        t_cold_start = time.perf_counter()
        if not local_clip_vision.is_loaded:
            local_clip_vision.load()
        cold_start_ms = round((time.perf_counter() - t_cold_start) * 1000, 2)

        # Synthetic 224x224 RGB image
        img = Image.new("RGB", (224, 224), color=(50, 120, 200))
        bio = io.BytesIO()
        img.save(bio, format="PNG")
        png_bytes = bio.getvalue()

        latencies_ms: List[float] = []
        cpu_readings: List[float] = []

        for i in range(self.iterations):
            self.process.cpu_percent(interval=None)
            t0 = time.perf_counter()
            _ = local_clip_vision.inspect_image(image_bytes=png_bytes, filename="bench.png")
            elapsed = (time.perf_counter() - t0) * 1000
            latencies_ms.append(elapsed)
            cpu_readings.append(self.process.cpu_percent(interval=None))

        mem_after = self.process.memory_info().rss / (1024 * 1024)
        stats = compute_statistics(latencies_ms)
        avg_cpu = round(sum(cpu_readings) / len(cpu_readings), 1) if cpu_readings else 0.0

        return {
            "model_name": local_clip_vision.model_name,
            "modality": "Vision Embedding & Visual Classification",
            "runtime": "ONNX Runtime (Real Neural Network)",
            "target_hardware": "Snapdragon X Elite Hexagon NPU",
            "execution_unit": f"{local_clip_vision.execution_provider} (Dev Host)",
            "target_quantization": "FP32 / QNN W8A16 Compatible",
            "active_quantization": "Standard ONNX Weights",
            "input_size": "224x224 RGB Image (150,528 pixels)",
            "iterations": self.iterations,
            "cold_start_ms": cold_start_ms,
            "latency_stats": stats,
            "rss_memory_mb": round(mem_after, 2),
            "memory_delta_mb": round(mem_after - mem_before, 2),
            "avg_cpu_percent": avg_cpu,
            "npu_usage": "N/A - Host architecture has no Hexagon NPU",
        }

    def benchmark_ocr_model(self) -> Dict[str, Any]:
        """Provides truthful status for EasyOCR model."""
        return {
            "model_name": "EasyOCR (CRAFT + CRNN)",
            "modality": "Optical Character Recognition (OCR)",
            "runtime": "ONNX Runtime (QNN)",
            "target_hardware": "Snapdragon X Elite Hexagon NPU",
            "execution_unit": "Not Yet Compiled (HTP context binary pending)",
            "target_quantization": "w8a8 / w8a16",
            "active_quantization": "N/A",
            "input_size": "Document page / scanned image",
            "iterations": 0,
            "cold_start_ms": 0.0,
            "latency_stats": {"min_ms": 0.0, "max_ms": 0.0, "mean_ms": 0.0, "median_ms": 0.0, "p95_ms": 0.0},
            "rss_memory_mb": 0.0,
            "memory_delta_mb": 0.0,
            "avg_cpu_percent": 0.0,
            "npu_usage": "N/A - Model offline compilation scheduled",
            "status_note": "Awaiting offline CRAFT/CRNN HTP compilation via Qualcomm AI Hub recipe.",
        }

    def run_all_benchmarks(self) -> Dict[str, Any]:
        """
        Executes full benchmark suite across all NEXUS models.
        Persists raw results to data/benchmarks/benchmark_results.json.
        """
        t_all_start = time.perf_counter()
        hw_profile = audit_hardware_and_providers()

        embed_res = self.benchmark_embeddings()
        slm_res = self.benchmark_language_model()
        speech_res = self.benchmark_speech_model()
        vision_res = self.benchmark_vision_model()
        ocr_res = self.benchmark_ocr_model()

        total_elapsed_sec = round(time.perf_counter() - t_all_start, 2)
        timestamp = datetime.now(timezone.utc).isoformat()

        models_list = [embed_res, slm_res, speech_res, vision_res, ocr_res]

        full_results: Dict[str, Any] = {
            "timestamp": timestamp,
            "total_benchmark_duration_seconds": total_elapsed_sec,
            "iterations_per_model": self.iterations,
            "hardware_profile": hw_profile,
            "models": models_list,
            "summary": {
                "fastest_warm_model": min(
                    [m for m in models_list if m["iterations"] > 0],
                    key=lambda m: m["latency_stats"]["median_ms"],
                )["model_name"],
                "total_models_evaluated": len(models_list),
                "active_models_measured": len([m for m in models_list if m["iterations"] > 0]),
            },
        }

        # Persist raw benchmark metrics to disk
        try:
            with open(BENCHMARKS_FILE, "w", encoding="utf-8") as f:
                json.dump(full_results, f, indent=2)
            logger.info(f"[BENCHMARK] Successfully persisted raw benchmark results to {BENCHMARKS_FILE}")
        except Exception as e:
            logger.error(f"[BENCHMARK] Failed to save benchmark results to {BENCHMARKS_FILE}: {e}")

        return full_results


def get_latest_benchmark_results() -> Optional[Dict[str, Any]]:
    """Retrieves the most recent benchmark run from disk or runs a baseline if missing."""
    if BENCHMARKS_FILE.exists():
        try:
            with open(BENCHMARKS_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Error reading benchmark results file: {e}")

    # If no benchmark has been executed yet, run a baseline 3-iteration suite
    runner = BenchmarkRunner(iterations=3)
    return runner.run_all_benchmarks()
