"""``model-lab`` command-line interface.

Exit codes: 0 success, 1 validation / lookup failure, 2 usage error (argparse).
No command downloads weights or writes outside paths you pass explicitly.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from collections.abc import Sequence
from pathlib import Path

from model_lab.registry.loader import (
    DEFAULT_REGISTRY,
    DEFAULT_SCHEMA,
    RegistryError,
    filter_models,
    get_model,
    json_schema,
    load_registry,
)
from model_lab.registry.schema import ModelEntry, Role, Task


def _enum_choices(enum: type[Task] | type[Role]) -> list[str]:
    return [e.value for e in enum]


def _row(m: ModelEntry) -> str:
    tasks = ",".join(t.value for t in m.tasks)
    lic, ver = m.licensing.commercial_use.value, m.verification.status.value
    return f"{m.id:<28} {m.role.value:<20} {lic:<22} {ver:<19} {tasks}"


def cmd_registry_list(args: argparse.Namespace) -> int:
    reg = load_registry(args.registry)
    models = filter_models(
        reg.models,
        task=Task(args.task) if args.task else None,
        role=Role(args.role) if args.role else None,
    )
    if args.json:
        print(json.dumps([m.model_dump(mode="json") for m in models], indent=2))
        return 0
    print(f"{'id':<28} {'role':<20} {'commercial_use':<22} {'verification':<19} tasks")
    for m in models:
        print(_row(m))
    print(
        f"\n{len(models)} of {len(reg.models)} entries (research baseline {reg.research_baseline})"
    )
    return 0


def cmd_registry_show(args: argparse.Namespace) -> int:
    reg = load_registry(args.registry)
    try:
        m = get_model(reg, args.model_id)
    except KeyError:
        print(f"unknown model id: {args.model_id}", file=sys.stderr)
        return 1
    print(json.dumps(m.model_dump(mode="json", exclude_none=True), indent=2))
    return 0


def cmd_registry_validate(args: argparse.Namespace) -> int:
    reg = load_registry(args.registry)  # raises RegistryError -> handled in main
    problems: list[str] = []
    for m in reg.models:
        if m.card and not (args.registry.parent / m.card).is_file():
            problems.append(f"{m.id}: card not found: {m.card}")
    if args.check_schema_file and DEFAULT_SCHEMA.is_file():
        on_disk = json.loads(DEFAULT_SCHEMA.read_text(encoding="utf-8"))
        if on_disk != json_schema():
            problems.append(
                f"{DEFAULT_SCHEMA.name} is stale: run `model-lab registry schema --write`"
            )
    for p in problems:
        print(f"ERROR {p}", file=sys.stderr)
    if problems:
        return 1
    print(f"OK: {len(reg.models)} entries valid in {args.registry}")
    return 0


def cmd_registry_schema(args: argparse.Namespace) -> int:
    text = json.dumps(json_schema(), indent=2, sort_keys=True) + "\n"
    if args.write:
        DEFAULT_SCHEMA.write_text(text, encoding="utf-8")
        print(f"wrote {DEFAULT_SCHEMA}")
    else:
        print(text, end="")
    return 0


def cmd_registry_audit(args: argparse.Namespace) -> int:
    """Summarise evidence quality: what is verified, gated, or missing an artifact."""
    reg = load_registry(args.registry)
    ver = Counter(m.verification.status.value for m in reg.models)
    lic = Counter(m.licensing.commercial_use.value for m in reg.models)
    print("verification status:", dict(ver))
    print("commercial use:     ", dict(lic))
    for kind in (
        "official_pretrained",
        "automotive_finetuned",
        "quantized",
        "onnx_export",
        "engine",
    ):
        c = Counter(getattr(m.artifacts, kind).status.value for m in reg.models)
        print(f"artifact {kind:<21}", dict(c))
    queue = [m.id for m in reg.models if m.verification.status.value != "verified_primary"]
    print(f"\nverification queue ({len(queue)}): {', '.join(queue)}")
    return 0


def cmd_env_check(args: argparse.Namespace) -> int:
    from model_lab.utilities.env import collect

    rep = collect(network=args.network)
    if args.json:
        print(json.dumps(rep.to_dict(), indent=2, default=str))
        return 0
    print(
        f"python {rep.python} ({'venv' if rep.executable_in_venv else 'system'}) on {rep.platform}"
    )
    print(f"cpus {rep.cpu_count}, RAM {rep.mem_total_gb} GB, free disk {rep.disk_free_gb} GB")
    print(f"torch cuda: {rep.gpu.torch_cuda_available}  devices: {rep.gpu.devices or '-'}")
    print(f"nvidia-smi: {rep.gpu.nvidia_smi or 'not found'}")
    installed = {k: v for k, v in rep.packages.items() if v}
    print("installed:", ", ".join(f"{k}=={v}" for k, v in installed.items()))
    print("capabilities:")
    for k, v in rep.capabilities.items():
        print(f"  [{'x' if v else ' '}] {k}")
    if rep.network is not None:
        print("network:")
        for url, status in rep.network.items():
            print(f"  {status:<30} {url}")
    return 0


def cmd_benchmark_synthetic(args: argparse.Namespace) -> int:
    """Benchmark the reference voxelizer on a synthetic scene (CPU smoke benchmark)."""
    from model_lab.benchmarking.harness import BenchmarkConfig, run_benchmark
    from model_lab.data.synthetic import make_scene
    from model_lab.data.voxelize import VoxelGrid, voxelize

    grid = VoxelGrid((0.16, 0.16, 4.0), (0.0, -39.68, -3.0, 69.12, 39.68, 1.0))
    scenes = [make_scene(seed=s) for s in range(4)]
    cfg = BenchmarkConfig(
        model_id="reference-numpy-voxelizer",
        input_desc=f"synthetic 32-beam scene, ~{len(scenes[0].points)} pts, pillar grid {grid.grid_size}",
        runtime="numpy",
        warmup=args.warmup,
        iters=args.iters,
    )
    rec = run_benchmark(
        cfg,
        make_input=lambda i: scenes[i % len(scenes)].points,
        stages={"voxelize": lambda p: voxelize(p, grid, max_points_per_voxel=32, max_voxels=16000)},
    )
    print(rec.to_json())
    if args.out:
        print(f"saved {rec.save(args.out)}", file=sys.stderr)
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="model-lab", description=__doc__.splitlines()[0] if __doc__ else None
    )
    sub = p.add_subparsers(dest="group", required=True)

    reg = sub.add_parser("registry", help="inspect and validate models/registry.yaml")
    reg.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    rsub = reg.add_subparsers(dest="cmd", required=True)
    ls = rsub.add_parser("list", help="table of entries")
    ls.add_argument("--task", choices=_enum_choices(Task))
    ls.add_argument("--role", choices=_enum_choices(Role))
    ls.add_argument("--json", action="store_true")
    ls.set_defaults(func=cmd_registry_list)
    show = rsub.add_parser("show", help="full entry as JSON")
    show.add_argument("model_id")
    show.set_defaults(func=cmd_registry_show)
    val = rsub.add_parser("validate", help="schema + cross-file checks")
    val.add_argument(
        "--check-schema-file", action="store_true", help="fail if registry.schema.json is stale"
    )
    val.set_defaults(func=cmd_registry_validate)
    sch = rsub.add_parser("schema", help="print or write the JSON Schema")
    sch.add_argument("--write", action="store_true")
    sch.set_defaults(func=cmd_registry_schema)
    aud = rsub.add_parser("audit", help="evidence / artifact status summary")
    aud.set_defaults(func=cmd_registry_audit)

    env = sub.add_parser("env", help="environment preflight")
    esub = env.add_subparsers(dest="cmd", required=True)
    chk = esub.add_parser("check", help="report Python, packages, GPU, capabilities (read-only)")
    chk.add_argument("--network", action="store_true", help="also HEAD-probe HF / GitHub / PyPI")
    chk.add_argument("--json", action="store_true")
    chk.set_defaults(func=cmd_env_check)

    bench = sub.add_parser("benchmark", help="provenance-recording benchmarks")
    bsub = bench.add_subparsers(dest="cmd", required=True)
    syn = bsub.add_parser("synthetic", help="CPU smoke benchmark of the reference voxelizer")
    syn.add_argument("--warmup", type=int, default=2)
    syn.add_argument("--iters", type=int, default=10)
    syn.add_argument(
        "--out", type=Path, help="directory for the JSON record (e.g. benchmarks/results)"
    )
    syn.set_defaults(func=cmd_benchmark_synthetic)
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        rc: int = args.func(args)
    except RegistryError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
