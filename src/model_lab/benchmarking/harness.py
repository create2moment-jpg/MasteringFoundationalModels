"""Latency / throughput benchmark harness with full provenance.

A latency number without its context is useless. Every :class:`BenchmarkRecord`
captures: model id and revision, dataset/split, code/config hash, hardware and
software versions, input shape, batch size, precision, warm-up and timed
iterations, per-stage latency percentiles (pre-process / forward /
post-process / end-to-end), throughput, peak memory where available and task
metrics.

The harness is framework-agnostic: stages are plain callables. For GPU work,
pass a ``sync`` callable (e.g. ``torch.cuda.synchronize``) – without it you
time kernel *launches*, not execution, and numbers look impossibly good.
"""

from __future__ import annotations

import hashlib
import json
import platform
import statistics
import subprocess
import time
import tracemalloc
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

Stage = Callable[[Any], Any]


@dataclass(frozen=True)
class LatencyStats:
    mean_ms: float
    p50_ms: float
    p90_ms: float
    p99_ms: float
    min_ms: float
    max_ms: float

    @staticmethod
    def from_samples(samples_s: list[float]) -> LatencyStats:
        ms = sorted(s * 1000 for s in samples_s)

        def pct(q: float) -> float:
            # nearest-rank percentile: stable for small n, no interpolation surprises
            k = max(0, min(len(ms) - 1, round(q / 100 * len(ms) + 0.5) - 1))
            return ms[k]

        return LatencyStats(
            mean_ms=statistics.fmean(ms),
            p50_ms=pct(50),
            p90_ms=pct(90),
            p99_ms=pct(99),
            min_ms=ms[0],
            max_ms=ms[-1],
        )


@dataclass
class BenchmarkConfig:
    model_id: str
    model_revision: str | None = None
    dataset: str = "synthetic"
    split: str = "n/a"
    input_desc: str = ""  # e.g. "1x3x640x640" or "~28k points, 1 sweep"
    batch_size: int = 1
    precision: str = "fp32"
    runtime: str = "python"
    warmup: int = 5
    iters: int = 30
    extra: dict[str, Any] = field(default_factory=dict)

    def config_hash(self) -> str:
        blob = json.dumps(asdict(self), sort_keys=True, default=str).encode()
        return hashlib.sha256(blob).hexdigest()[:16]


@dataclass
class BenchmarkRecord:
    config: BenchmarkConfig
    config_hash: str
    git_commit: str | None
    environment: dict[str, Any]
    stages: dict[str, LatencyStats]
    end_to_end: LatencyStats
    throughput_items_per_s: float
    peak_python_mem_mb: float | None
    metrics: dict[str, float]
    checkpoint_source: str | None
    timestamp_utc: str

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, sort_keys=True, default=str)

    def save(self, directory: Path) -> Path:
        directory.mkdir(parents=True, exist_ok=True)
        stamp = self.timestamp_utc.replace(":", "").replace("-", "")
        path = directory / f"{self.config.model_id}_{self.config_hash}_{stamp}.json"
        path.write_text(self.to_json() + "\n", encoding="utf-8")
        return path


def _git_commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
        dirty = subprocess.run(
            ["git", "status", "--porcelain"], capture_output=True, text=True, timeout=5
        )
        return out.stdout.strip() + ("-dirty" if dirty.stdout.strip() else "")
    except (OSError, subprocess.SubprocessError):
        return None


def basic_environment() -> dict[str, Any]:
    """Minimal, privacy-safe environment description (no hostnames, users or paths)."""
    from model_lab.utilities.env import package_versions

    return {
        "python": platform.python_version(),
        "platform": f"{platform.system()} {platform.machine()}",
        "processor": platform.processor() or None,
        "packages": package_versions(),
    }


def run_benchmark(
    config: BenchmarkConfig,
    make_input: Callable[[int], Any],
    stages: Mapping[str, Stage],
    *,
    sync: Callable[[], None] | None = None,
    metrics: Callable[[], dict[str, float]] | None = None,
    checkpoint_source: str | None = None,
    track_python_memory: bool = False,
) -> BenchmarkRecord:
    """Time a pipeline of stages ``input -> stage_1 -> stage_2 -> ...``.

    Args:
        config: what is being measured; serialised into the record.
        make_input: ``i -> input`` for iteration ``i``; called *outside* the timed region.
        stages: ordered mapping, e.g. ``{"preprocess": f, "forward": g, "postprocess": h}``.
        sync: device barrier called before each clock read (required for async devices).
        metrics: optional callable returning task metrics (computed once, untimed).
        checkpoint_source: URL / path / hash of the weights actually used.
        track_python_memory: use tracemalloc (adds overhead; Python heap only, not GPU).
    """

    def barrier() -> None:
        if sync is not None:
            sync()

    for i in range(config.warmup):
        x = make_input(i)
        for fn in stages.values():
            x = fn(x)
    barrier()

    per_stage: dict[str, list[float]] = {name: [] for name in stages}
    e2e: list[float] = []
    if track_python_memory:
        tracemalloc.start()
    for i in range(config.iters):
        x = make_input(config.warmup + i)
        barrier()
        t_start = time.perf_counter()
        for name, fn in stages.items():
            t0 = time.perf_counter()
            x = fn(x)
            barrier()
            per_stage[name].append(time.perf_counter() - t0)
        e2e.append(time.perf_counter() - t_start)
    peak_mb = None
    if track_python_memory:
        peak_mb = tracemalloc.get_traced_memory()[1] / 2**20
        tracemalloc.stop()

    e2e_stats = LatencyStats.from_samples(e2e)
    return BenchmarkRecord(
        config=config,
        config_hash=config.config_hash(),
        git_commit=_git_commit(),
        environment=basic_environment(),
        stages={k: LatencyStats.from_samples(v) for k, v in per_stage.items()},
        end_to_end=e2e_stats,
        throughput_items_per_s=config.batch_size * 1000.0 / e2e_stats.mean_ms,
        peak_python_mem_mb=peak_mb,
        metrics=metrics() if metrics else {},
        checkpoint_source=checkpoint_source,
        timestamp_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    )
