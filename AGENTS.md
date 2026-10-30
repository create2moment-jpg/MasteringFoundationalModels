# Instructions for AI coding agents

Scope: **autonomous-driving perception only** (2D/3D detection, encoders, BEV, temporal,
camera-LiDAR fusion, and the fine-tuning, quantisation and deployment of those). Do not add
general LLM, indoor-robotics, planning or control content.

## Before changing anything

```bash
source .venv/bin/activate || (uv venv && uv pip install -e ".[dev,export]")
pytest -q && ruff check . && ruff format --check . && mypy
model-lab registry validate --check-schema-file
```

## Hard rules

- Never invent a model name, URL, licence, score, VRAM figure or latency. Use `null` / `not_verified` and add the item to `docs/sources/README.md#verification-queue`.
- Registry edits must update the `verification` block (date, status, method, evidence with `source_type`).
- `claim_type: reproduced_local` requires a committed record in `benchmarks/results/` (a test enforces this).
- If you change `src/model_lab/registry/schema.py`, run `model-lab registry schema --write`.
- Never commit weights, datasets, engines, credentials, or outputs with hostnames, usernames or paths.
- Label scripts you could not run as **NOT EXECUTED** in their docstring and README.
- Keep core dependencies to numpy, pydantic and pyyaml. Heavy stacks go in extras or external installs.

## Conventions

- Box convention: `[x, y, z, l, w, h, yaw]`, gravity centre, LiDAR frame (x fwd, y left, z up). See `src/model_lab/data/geometry.py`.
- Public functions document shapes, units and frames.
- Tests use synthetic fixtures (`model_lab.data.synthetic`). Mark tests that need network, GPU or weights with `@pytest.mark.integration` / `@pytest.mark.gpu`.
