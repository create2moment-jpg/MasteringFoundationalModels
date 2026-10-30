import json
import time

import pytest

from model_lab.benchmarking.harness import BenchmarkConfig, LatencyStats, run_benchmark
from model_lab.cli import main


def test_latency_stats_nearest_rank() -> None:
    s = LatencyStats.from_samples([i / 1000 for i in range(1, 101)])  # 1..100 ms
    assert s.p50_ms == pytest.approx(50)
    assert s.p90_ms == pytest.approx(90)
    assert s.min_ms == pytest.approx(1) and s.max_ms == pytest.approx(100)


def test_run_benchmark_records_stages_and_provenance(tmp_path) -> None:
    calls = {"sync": 0}

    def sync() -> None:
        calls["sync"] += 1

    cfg = BenchmarkConfig(model_id="toy", warmup=2, iters=5)
    rec = run_benchmark(
        cfg,
        make_input=lambda i: i,
        stages={"pre": lambda x: x + 1, "forward": lambda x: (time.sleep(0.001), x)[1]},
        sync=sync,
        metrics=lambda: {"ap": 0.5},
    )
    assert set(rec.stages) == {"pre", "forward"}
    assert rec.stages["forward"].p50_ms >= 0.9
    assert rec.end_to_end.mean_ms >= rec.stages["forward"].mean_ms
    assert rec.metrics == {"ap": 0.5}
    assert calls["sync"] >= 5
    assert rec.config_hash == cfg.config_hash()
    path = rec.save(tmp_path)
    assert json.loads(path.read_text())["config"]["model_id"] == "toy"


def test_cli_registry_commands(capsys) -> None:
    assert main(["registry", "validate", "--check-schema-file"]) == 0
    assert main(["registry", "list", "--task", "lidar_3d_detection"]) == 0
    out = capsys.readouterr().out
    assert "pointpillars-openpcdet" in out and "dinov2" not in out
    assert main(["registry", "show", "utonia"]) == 0
    assert json.loads(capsys.readouterr().out)["licensing"]["commercial_use"] == "non_commercial"
    assert main(["registry", "show", "nope"]) == 1


def test_cli_env_check_json(capsys) -> None:
    assert main(["env", "check", "--json"]) == 0
    rep = json.loads(capsys.readouterr().out)
    assert rep["capabilities"]["registry_and_cpu_labs"] is True
    assert rep["network"] is None  # network probing is opt-in


def test_cli_rejects_bad_task() -> None:
    with pytest.raises(SystemExit) as exc:
        main(["registry", "list", "--task", "planning"])
    assert exc.value.code == 2
