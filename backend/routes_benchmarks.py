"""
NEXUS Phase 10: Benchmark & Snapdragon Optimization API Routes
Provides endpoints for querying system hardware profile, execution providers,
retrieving empirical performance benchmarks, and executing live benchmark sweeps.
"""

from typing import Any, Dict, Optional
from fastapi import APIRouter, HTTPException, Query, status
from pydantic import BaseModel, Field

from backend.logger import get_logger
from backend.optimization.benchmarks import (
    BenchmarkRunner,
    audit_hardware_and_providers,
    get_latest_benchmark_results,
)

logger = get_logger("nexus.routes_benchmarks")

benchmarks_router = APIRouter(prefix="/benchmarks", tags=["Snapdragon Optimization & Performance"])


class BenchmarkRunRequest(BaseModel):
    iterations: int = Field(default=3, ge=1, le=20, description="Warm inference iterations per model")


@benchmarks_router.get(
    "/status",
    summary="Get Hardware Profile & Execution Providers",
    description="Audits CPU silicon, available memory, ONNX Runtime execution providers, and Qualcomm Hexagon NPU presence.",
)
def get_hardware_status() -> Dict[str, Any]:
    return audit_hardware_and_providers()


@benchmarks_router.get(
    "/latest",
    summary="Get Latest Benchmark Metrics",
    description="Retrieves the most recent empirical benchmark run from disk or runs a baseline measurement.",
)
def get_latest_benchmarks() -> Dict[str, Any]:
    results = get_latest_benchmark_results()
    if not results:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No benchmark results available yet.",
        )
    return results


@benchmarks_router.post(
    "/run",
    summary="Trigger Live Performance Benchmark",
    description="Executes empirical multi-iteration performance benchmarking across all local models and persists metrics to disk.",
)
def run_live_benchmark(request: Optional[BenchmarkRunRequest] = None) -> Dict[str, Any]:
    iterations = request.iterations if request else 3
    logger.info(f"[API_BENCHMARK] Triggering live performance benchmark ({iterations} iterations)...")
    try:
        runner = BenchmarkRunner(iterations=iterations)
        results = runner.run_all_benchmarks()
        return results
    except Exception as e:
        logger.error(f"[API_BENCHMARK] Benchmark execution failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Benchmark execution failed: {str(e)}",
        )
