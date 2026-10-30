# 08: Licensing and data governance

> Not legal advice. The registry flags anything uncertain as `unclear_legal_review`.

## Three licences per model, always separate

1. **Code licence** (repository): Apache-2.0 or MIT is common.
2. **Weight licence** (checkpoint): may differ from the code. Examples in the registry: CC-BY-NC-4.0 (Utonia, Concerto), the DINOv3 custom licence (gated), the NVIDIA Open Model License (C-RADIOv3), and AGPL-3.0 (Ultralytics).
3. **Dataset terms** (what the weights were trained on): KITTI (CC BY-NC-SA 3.0), nuScenes (non-commercial unless commercially licensed), Waymo (restrictive; many projects cannot redistribute Waymo-trained weights; for example, the Pointcept PTv3 card says the Waymo weights are not released).

Permissive code with non-commercial weights or data gives you **a research artifact**, not a
product component. Open-source code never implies open weights.

## Registry encoding

`licensing.commercial_use` is one of:
- `permissive`: weights under Apache/MIT-style terms (dataset terms may still apply)
- `conditional`: custom licence allowing commercial use with conditions (read it)
- `non_commercial`
- `copyleft`: e.g. AGPL obligations
- `unclear_legal_review`: default whenever any of the three layers is unknown or restrictive

Current distribution: `model-lab registry audit`.

## Gated models

Gated repositories (e.g. DINOv3 on HF) require accepting terms with a personal account. Do not
share tokens and do not mirror gated weights internally without approval.

## Data handling rules for this repo

- Never commit datasets, weights, engines, calibration sets, or images from proprietary fleets.
- Keep company data inside approved environments. Do not upload it to external services, including hosted notebooks and model hubs.
- Benchmark records contain no hostnames, usernames or absolute paths (see `basic_environment`).
- Synthetic fixtures (`model_lab.data.synthetic`) exist so tests and CI never need real data.
