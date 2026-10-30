"""Provenance-preserving latency / throughput benchmarking."""

from model_lab.benchmarking.harness import (
    BenchmarkConfig,
    BenchmarkRecord,
    LatencyStats,
    run_benchmark,
)

__all__ = ["BenchmarkConfig", "BenchmarkRecord", "LatencyStats", "run_benchmark"]
