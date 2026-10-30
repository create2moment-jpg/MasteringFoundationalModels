# Autonomous-Driving Perception Foundation-Model Mastery Lab

A living, evidence-backed reference and lab environment for **selecting, understanding,
adapting, evaluating, compressing and deploying** pretrained and foundation models for
autonomous-driving perception: 2D camera detection, image encoders, LiDAR 3D detection,
point-cloud encoders, camera-only 3D and temporal perception, and camera-LiDAR fusion.

> Research baseline **2026-10-30**. Fast-moving facts (licences, checkpoints, engines) carry a
> `last_verified` date and a verification status. Re-verify before relying on them.

## Quick start

```bash
uv venv && uv pip install -e ".[dev,export]"       # or python -m venv .venv && pip install -e ".[dev,export]"
source .venv/bin/activate

model-lab env check                                 # what can this machine run?
model-lab registry validate --check-schema-file     # schema + cards + JSON-schema freshness
model-lab registry list --task lidar_3d_detection   # shortlist by task
model-lab registry show bevfusion-mit               # full evidence-backed entry
model-lab registry audit                            # verified vs unverified, artifact availability

pytest && ruff check . && ruff format --check . && mypy

python labs/04_point_cloud_basics/explore_scene.py      # LiDAR geometry intuition (CPU)
python labs/05_lidar_3d_baseline/run_cluster_baseline.py  # 3D metrics anatomy (CPU)
python labs/09_quantization_and_export/onnx_parity_toy.py # export/INT8 parity loop (CPU)
```

## Repository map

| path | what |
|---|---|
| [`docs/`](docs/) | 00 vision/roadmap · 01 taxonomy · 02 selection playbook · 03 12-week path · 04 fine-tuning · 05 deployment · 06 geometry & evaluation · 07 reproducibility · 08 licensing · 09 glossary · `decisions/` ADRs · `sources/` source index + verification queue |
| [`models/registry.yaml`](models/registry.yaml) | 23 curated entries with licences, I/O contracts, artifact status and evidence ([schema](src/model_lab/registry/schema.py), [JSON Schema](models/registry.schema.json)) |
| [`models/cards/`](models/cards/) | reference cards: DINOv2 (encoder), RT-DETRv2 (2D), Utonia (point encoder + domain fit), CenterPoint (LiDAR 3D), BEVFusion (fusion + deployment path) |
| [`labs/`](labs/) | 00 env/registry · 01 2D baseline · 04 point-cloud basics · 05 LiDAR baseline · 09 export & quantisation |
| [`recipes/`](recipes/) | 2D fine-tuning · image-encoder adaptation strategies · LiDAR (OpenPCDet) · deployment (ONNX → TensorRT + parity) |
| [`src/model_lab/`](src/model_lab/) | `registry` · `data` (frames, boxes, projection, voxelisation, synthetic LiDAR) · `evaluation` (IoU, NMS, AP variants) · `benchmarking` (provenance-recording harness) · `baselines` · `utilities` (env preflight, seeding) · `cli` |
| [`configs/`](configs/) | class maps (COCO → driving) |
| [`benchmarks/`](benchmarks/) | record template; `results/` for records that back registry claims |

## Recommendations at a glance (details and caveats in [docs/02](docs/02_model_selection_playbook.md))

| task | start here | why / caveat |
|---|---|---|
| 2D driving detector | `rt-detrv2-r50vd` → fine-tune on nuImages/BDD100K | Apache-2.0, NMS-free, clean export. No driving numbers in the registry yet |
| image encoder | `dinov2-vitb14` (permissive) vs `c-radiov3-b` (high-res, NVIDIA OML) vs `dinov3-vitb16` (gated, custom licence) | **no verified winner** on driving data |
| open-vocabulary | `grounding-dino-base`, `owlv2-base-p16-ensemble` | offline auto-labelling, not on-vehicle |
| point-cloud encoder | `utonia` (outdoor LiDAR in pretraining) | **CC-BY-NC**; no detection head; transfer to 3D detection unproven |
| LiDAR 3D | `centerpoint-nuscenes`, `pointpillars-openpcdet` | weights inherit dataset terms (non-commercial) |
| camera-only 3D | `sparse4d-v3`, `streampetr`, `bevformer` | mostly unverified this session |
| fusion | `bevfusion-mit` | most complete public deployment path (Autoware ONNX → TensorRT) |

## What was executed and what was not (2026-10-30 build)

- **Executed** (CPU-only container, Python 3.13, no GPU, no torch): unit tests, lint, format, type-check, CLI commands, Labs 04, 05 A and 09 A, and the synthetic benchmark.
- **Written, not executed:** `labs/01_2d_baseline/infer_hf.py`, `recipes/image_encoders/finetune_encoder.py`, `recipes/deployment/export_rtdetr_parity.py` (need torch/GPU/Hub downloads), and all OpenPCDet/TensorRT commands.
- **Research access:** the Hugging Face Hub was reachable via a connector, so model cards and metadata were verified there. github.com, arxiv.org and vendor docs were **not** reachable, so GitHub-hosted facts are marked secondary or prior knowledge. See [`docs/sources/`](docs/sources/README.md).

## Principles

1. Separate *paper claim*, *repo-reported*, *locally reproduced* and *production suitability*.
2. Keep checkpoint ≠ quantised weights ≠ ONNX ≠ engine ≠ runtime.
3. Code licence ≠ weight licence ≠ dataset terms.
4. Generic benchmarks are pretraining context, never driving evidence.
5. When evidence is missing, say `not_verified`, never guess.

See [CONTRIBUTING.md](CONTRIBUTING.md) for the update process and [AGENTS.md](AGENTS.md) for
instructions to AI coding agents working in this repo.
