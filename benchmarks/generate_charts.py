"""
NEXUS Benchmark Chart Generator (Phase 13)
Generates high-resolution visualization figures from benchmark_summary.json.
Outputs saved to benchmarks/charts/:
1. latency_comparison.png
2. memory_rss_profile.png
3. workflow_breakdown.png
4. cold_vs_warm.png
"""

import os
import sys
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")  # Headless rendering
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

BENCHMARKS_DIR = PROJECT_ROOT / "benchmarks"
RESULTS_FILE = BENCHMARKS_DIR / "results" / "benchmark_summary.json"
CHARTS_DIR = BENCHMARKS_DIR / "charts"
CHARTS_DIR.mkdir(parents=True, exist_ok=True)


def set_plot_theme():
    """Sets a polished dark-mode styling aligned with the NEXUS palette."""
    plt.rcParams.update({
        "figure.facecolor": "#080b12",
        "axes.facecolor": "#0e1322",
        "axes.edgecolor": "#334155",
        "axes.labelcolor": "#94a3b8",
        "xtick.color": "#94a3b8",
        "ytick.color": "#94a3b8",
        "text.color": "#f8fafc",
        "grid.color": "#1e293b",
        "grid.linestyle": "--",
        "font.family": "sans-serif",
        "font.sans-serif": ["Segoe UI", "Helvetica", "Arial", "DejaVu Sans"],
    })


def generate_all_charts():
    if not RESULTS_FILE.exists():
        print(f"Error: {RESULTS_FILE} not found. Run benchmarks/suite.py first.")
        sys.exit(1)

    with open(RESULTS_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    results = data.get("results", [])
    set_plot_theme()

    # Filter components with valid latency measurements
    measured = [r for r in results if r.get("median_latency_ms") is not None]

    labels = [r["component"] for r in measured]
    medians = [r["median_latency_ms"] for r in measured]
    p95s = [r["p95_latency_ms"] for r in measured]
    colds = [r["cold_latency_ms"] if r["cold_latency_ms"] is not None else 0.0 for r in measured]
    memories = [r["memory_rss_mb"] for r in results]
    all_labels = [r["component"] for r in results]

    # -------------------------------------------------------------------------
    # Chart 1: Latency Comparison (Median vs P95)
    # -------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 5.5), dpi=300)
    x = np.arange(len(labels))
    width = 0.35

    rects1 = ax.bar(x - width / 2, medians, width, label="Warm Median (ms)", color="#00e5ff", edgecolor="none")
    rects2 = ax.bar(x + width / 2, p95s, width, label="Warm P95 (ms)", color="#a855f7", edgecolor="none")

    ax.set_ylabel("Latency (milliseconds, log scale)", fontsize=11, fontweight="bold")
    ax.set_title("NEXUS Component Latency Profile (Median vs P95)", fontsize=13, fontweight="bold", pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right", fontsize=9, fontweight="bold")
    ax.set_yscale("log")
    ax.legend(frameon=True, facecolor="#162038", edgecolor="none", fontsize=10)
    ax.grid(True, which="both", axis="y", alpha=0.5)

    # Data value labels on top of bars
    for rect in rects1:
        h = rect.get_height()
        ax.annotate(f"{h:.1f}ms",
                    xy=(rect.get_x() + rect.get_width() / 2, h),
                    xytext=(0, 4), textcoords="offset points",
                    ha="center", va="bottom", fontsize=7.5, color="#00e5ff", fontweight="bold")

    for rect in rects2:
        h = rect.get_height()
        ax.annotate(f"{h:.1f}ms",
                    xy=(rect.get_x() + rect.get_width() / 2, h),
                    xytext=(0, 4), textcoords="offset points",
                    ha="center", va="bottom", fontsize=7.5, color="#c084fc", fontweight="bold")

    fig.tight_layout()
    chart1_path = CHARTS_DIR / "latency_comparison.png"
    fig.savefig(chart1_path)
    plt.close(fig)
    print(f"Generated chart: {chart1_path}")

    # -------------------------------------------------------------------------
    # Chart 2: Memory Footprint Profile (RSS MB)
    # -------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
    colors = ["#10b981" if r.get("status") == "VERIFIED" else "#f59e0b" for r in results]
    bars = ax.barh(all_labels, memories, color=colors, height=0.5)

    ax.set_xlabel("Process Resident Set Size (RSS in MB)", fontsize=11, fontweight="bold")
    ax.set_title("NEXUS Process Memory Footprint per Workload", fontsize=13, fontweight="bold", pad=15)
    ax.grid(True, axis="x", alpha=0.5)

    for bar in bars:
        w = bar.get_width()
        ax.annotate(f"{w:.1f} MB",
                    xy=(w, bar.get_y() + bar.get_height() / 2),
                    xytext=(6, 0), textcoords="offset points",
                    ha="left", va="center", fontsize=8.5, color="#f8fafc", fontweight="bold")

    fig.tight_layout()
    chart2_path = CHARTS_DIR / "memory_rss_profile.png"
    fig.savefig(chart2_path)
    plt.close(fig)
    print(f"Generated chart: {chart2_path}")

    # -------------------------------------------------------------------------
    # Chart 3: Complete Agent Workflow Step Breakdown
    # -------------------------------------------------------------------------
    workflow_res = next((r for r in results if r["component"] == "Complete Agent Workflow"), None)
    if workflow_res and "step_medians_ms" in workflow_res:
        step_dict = workflow_res["step_medians_ms"]
        clean_names = {
            "ingestion_ms": "1. Ingestion & Chunking",
            "indexing_ms": "2. Vector Indexing",
            "planning_ms": "3. Goal Planning (DAG)",
            "tool_execution_ms": "4. Tool Execution & Export",
            "verification_ms": "5. Evidence Verification",
        }

        step_labels = [clean_names.get(k, k) for k in step_dict.keys()]
        step_vals = list(step_dict.values())

        fig, ax = plt.subplots(figsize=(9, 5.2), dpi=300)
        palette = ["#00e5ff", "#3b82f6", "#8b5cf6", "#ec4899", "#10b981"]

        wedges, texts, autotexts = ax.pie(
            step_vals,
            labels=step_labels,
            autopct=lambda pct: f"{pct:.1f}%\n({pct * sum(step_vals) / 100:.1f}ms)",
            colors=palette[:len(step_vals)],
            startangle=140,
            textprops=dict(color="#f8fafc", fontsize=8.5, fontweight="bold"),
            wedgeprops=dict(width=0.6, edgecolor="#080b12", linewidth=2),
        )

        ax.set_title(f"Complete Agent Workflow Latency Distribution\n(Total Warm Median: {workflow_res['median_latency_ms']} ms)",
                     fontsize=12, fontweight="bold", pad=15)
        fig.tight_layout()
        chart3_path = CHARTS_DIR / "workflow_breakdown.png"
        fig.savefig(chart3_path)
        plt.close(fig)
        print(f"Generated chart: {chart3_path}")

    # -------------------------------------------------------------------------
    # Chart 4: Cold Start vs Warm Median Latency
    # -------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 5.5), dpi=300)
    x = np.arange(len(labels))
    width = 0.35

    rects1 = ax.bar(x - width / 2, colds, width, label="Cold Latency (ms)", color="#f59e0b", edgecolor="none")
    rects2 = ax.bar(x + width / 2, medians, width, label="Warm Median (ms)", color="#10b981", edgecolor="none")

    ax.set_ylabel("Latency (milliseconds, log scale)", fontsize=11, fontweight="bold")
    ax.set_title("Cold Start vs Warm Inference Acceleration", fontsize=13, fontweight="bold", pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right", fontsize=9, fontweight="bold")
    ax.set_yscale("log")
    ax.legend(frameon=True, facecolor="#162038", edgecolor="none", fontsize=10)
    ax.grid(True, which="both", axis="y", alpha=0.5)

    for rect in rects1:
        h = rect.get_height()
        if h > 0:
            ax.annotate(f"{h:.1f}ms",
                        xy=(rect.get_x() + rect.get_width() / 2, h),
                        xytext=(0, 4), textcoords="offset points",
                        ha="center", va="bottom", fontsize=7.5, color="#fbbf24", fontweight="bold")

    for rect in rects2:
        h = rect.get_height()
        if h > 0:
            ax.annotate(f"{h:.1f}ms",
                        xy=(rect.get_x() + rect.get_width() / 2, h),
                        xytext=(0, 4), textcoords="offset points",
                        ha="center", va="bottom", fontsize=7.5, color="#34d399", fontweight="bold")

    fig.tight_layout()
    chart4_path = CHARTS_DIR / "cold_vs_warm.png"
    fig.savefig(chart4_path)
    plt.close(fig)
    print(f"Generated chart: {chart4_path}")


if __name__ == "__main__":
    generate_all_charts()
