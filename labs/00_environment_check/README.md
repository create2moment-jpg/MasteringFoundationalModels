# Lab 00: environment and registry inspection

**Objective:** know what your machine can run, and learn to read the registry critically
before trusting any model reference.

## Commands (all read-only; nothing is downloaded)

```bash
uv venv && uv pip install -e ".[dev]"          # or: python -m venv .venv && pip install -e ".[dev]"
model-lab env check                             # Python, packages, GPU, capability flags
model-lab env check --network                   # + HEAD probes to HF / GitHub / PyPI (opt-in)
model-lab registry validate --check-schema-file
model-lab registry list --task lidar_3d_detection
model-lab registry show utonia
model-lab registry audit                        # verification + artifact-status summary
model-lab benchmark synthetic --iters 10        # provenance-recording CPU micro-benchmark
```

## Expected output (from the build container, 2026-10-30)

- `env check`: Python 3.13, 4 CPUs, no CUDA, `torch` absent. Only `registry_and_cpu_labs` (plus `onnx_export_and_parity` = false because torch is missing) are available.
- `registry audit`: 4 `verified_primary`, 15 `partially_verified`, 4 `not_verified`. Almost all quantised or engine artifacts are `not_verified` or `requires_local_build`.

## Questions

1. Pick three entries with `commercial_use: unclear_legal_review`. For each, which document would resolve the question: the code licence, the weight licence, or the dataset terms?
2. `bevfusion-mit` has an `onnx_export` marked `available_community`. Whose model is that ONNX, and what classes does it predict? Why does that matter for comparing metrics?
3. Why does the schema refuse `status: available_official` without a URL, yet allow `not_verified` without one?

## Why it works / when it fails

The registry is only as good as its evidence. `partially_verified` means *some* fields were
checked against a primary source. Read `verification.method` before relying on any field.

**Stretch:** pick one `not_verified` entry, verify it against primary sources, and update it
following [CONTRIBUTING.md](../../CONTRIBUTING.md).
