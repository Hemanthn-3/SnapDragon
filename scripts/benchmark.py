"""
NEXUS Snapdragon Benchmark CLI Runner
Executes multi-iteration empirical performance measurements across all local AI models.
Outputs formatted terminal telemetry and saves raw JSON metrics.
Usage:
    python scripts/benchmark.py [--iterations 5]
"""

import argparse
import json
import sys
from pathlib import Path

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.optimization.benchmarks import BenchmarkRunner, BENCHMARKS_FILE


def format_table(results: dict):
    hw = results["hardware_profile"]
    models = results["models"]

    print("\n" + "=" * 80)
    print("  NEXUS: SNAPDRAGON SILICON OPTIMIZATION & PERFORMANCE AUDIT")
    print("=" * 80)
    print(f"  Timestamp:         {results['timestamp']}")
    print(f"  Host Architecture: {hw['host_architecture']} ({hw['operating_system']})")
    print(f"  Processor:         {hw['processor_name']} ({hw['physical_cpu_cores']}C / {hw['logical_cpu_cores']}T)")
    print(f"  Memory:            {hw['available_ram_gb']} GB available / {hw['total_ram_gb']} GB total")
    print(f"  Target Silicon:    {hw['target_snapdragon_silicon']}")
    print(f"  Available ORT EPs: {', '.join(hw['onnx_available_providers'])}")
    print(f"  Hexagon NPU State: {hw['npu_status']}")
    print("-" * 80)
    print(f"  Iterations Tested: {results['iterations_per_model']} iterations per model")
    print(f"  Total Duration:    {results['total_benchmark_duration_seconds']}s")
    print("=" * 80)

    header = f"{'Model':<26} | {'Runtime':<18} | {'Quant':<10} | {'Cold(ms)':<8} | {'Median(ms)':<10} | {'P95(ms)':<8} | {'RSS(MB)':<7}"
    print(header)
    print("-" * len(header))

    for m in models:
        name = m["model_name"]
        if len(name) > 25:
            name = name[:23] + ".."
        runtime = m["runtime"]
        if len(runtime) > 17:
            runtime = runtime[:15] + ".."
        quant = m["target_quantization"]
        if len(quant) > 9:
            quant = quant[:8] + "."
        cold = f"{m['cold_start_ms']:.1f}"
        stats = m["latency_stats"]
        med = f"{stats['median_ms']:.1f}" if m["iterations"] > 0 else "N/A"
        p95 = f"{stats['p95_ms']:.1f}" if m["iterations"] > 0 else "N/A"
        rss = f"{m['rss_memory_mb']:.1f}" if m["rss_memory_mb"] > 0 else "N/A"

        row = f"{name:<26} | {runtime:<18} | {quant:<10} | {cold:<8} | {med:<10} | {p95:<8} | {rss:<7}"
        print(row)

    print("=" * 80)
    print(f"\n[OK] Raw benchmark metrics persisted to:\n  {BENCHMARKS_FILE.resolve()}\n")


def main():
    parser = argparse.ArgumentParser(description="NEXUS Local Performance Benchmark Runner")
    parser.add_argument(
        "--iterations",
        type=int,
        default=5,
        help="Number of warm inference iterations per model (default: 5)",
    )
    args = parser.parse_args()

    print(f"\n[INFO] Initializing NEXUS benchmark suite ({args.iterations} iterations per model)...")
    runner = BenchmarkRunner(iterations=args.iterations)
    results = runner.run_all_benchmarks()
    format_table(results)


if __name__ == "__main__":
    main()
