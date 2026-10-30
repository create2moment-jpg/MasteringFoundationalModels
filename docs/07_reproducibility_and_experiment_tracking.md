# 07: Reproducibility and experiment tracking

## Minimum record per run

| item | how |
|---|---|
| code | git commit (+ `-dirty` flag), recorded automatically by `BenchmarkRecord` |
| config | full resolved config saved next to outputs; config hash |
| data | manifest file + SHA-256; split and class map version |
| weights | source URL + revision / commit SHA + file hash |
| environment | `model-lab env check --json > env.json` (no user or host info) |
| seeds | `model_lab.utilities.seed.seed_everything(seed, deterministic_torch=...)` |
| results | metrics JSON + benchmark JSON in `benchmarks/results/` (committed only when they back a registry claim) |

## Determinism, honestly

Bit-exact GPU reproducibility is often impossible. Scatter-add (pillar scatter), atomics in
sparse conv, and cuDNN autotuning are nondeterministic. Use `deterministic_torch=True` to
*find* those ops, then accept the variance and **report the mean ± std over ≥ 3 seeds**
for any claimed improvement.

## Ablation design

- Change one factor at a time and keep the schedule in epochs *and* steps comparable.
- Use the same val split, resolution, threshold and class map.
- Pre-register the metric and the minimum meaningful effect, for example "+1.0 mAP at 40–80 m".
- Keep a lineage of checkpoint → parent checkpoint → data manifest.

## Tracking tools

Any tracker works (MLflow, Weights & Biases, TensorBoard, plain JSON). The repo is
tracker-agnostic: the `BenchmarkRecord` JSON is the portable unit. Do not commit tracker
artifacts that contain usernames, hostnames or absolute paths.

## Updating registry claims from your runs

1. Save the result JSON under `benchmarks/results/`.
2. Add a `benchmarks` entry with `claim_type: reproduced_local` and `notes: <result filename>`. A test enforces that the file exists.
3. Never overwrite a `reported_*` entry with a local number. Add a new entry alongside it.
