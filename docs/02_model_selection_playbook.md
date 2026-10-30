# 02: Model-selection playbook

"Best model" is conditional. Every recommendation below lists its **assumptions** and
**evidence level**, and "no verified winner" is an allowed answer. Evidence levels refer to
`verification.status` in `models/registry.yaml` at the 2026-10-30 baseline.

## Evidence rules

1. A paper's number is a `reported_paper` claim. A model-zoo table entry is `reported_repo`. Only a run with a committed record in `benchmarks/results/` is `reproduced_local`.
2. Compare numbers only under the same dataset, split, class map, metric implementation, input resolution and number of sweeps or frames. Otherwise, say "not comparable".
3. Generic benchmarks (ImageNet, COCO, ScanNet) provide **pretraining context only**. They are never evidence of driving performance.
4. Licence before accuracy. A model you cannot ship is a research baseline, not a candidate.

## Selection matrix

| need | candidate(s) | assumptions | evidence | gaps |
|---|---|---|---|---|
| Best reproducible driving 2D baseline | `rt-detrv2-r50vd` fine-tuned on nuImages/BDD100K | permissive licence required; GPU for training | HF metadata verified; no driving numbers in registry | run Lab 01 + recipe; record AP_S |
| Low-latency 2D | `yolo11-ultralytics` (AGPL) or a small RT-DETR variant | licence review done | not verified | measure on target HW with harness |
| Low-label 2D | frozen `dinov2-vitb14` / `c-radiov3-b` + DETR head | < 5k labelled frames | encoder metadata verified | Lab 02 strategy comparison not yet run |
| Open-vocabulary driving detector | `grounding-dino-base` (offline), `owlv2-base-p16-ensemble` | used for labelling, not on-vehicle | HF metadata verified | no driving-domain AP in registry |
| Image encoder for driving cameras | **no verified winner**: DINOv2 (permissive) vs DINOv3 (gated, custom licence) vs C-RADIOv3 (NVIDIA OML, high-res) | frozen-feature use | metadata/card verified | head-to-head on driving data needed |
| Pretrained point encoder with automotive transfer evidence | **no verified winner**. Best candidate `utonia` (outdoor LiDAR in pretraining, **NC licence**); `ptv3-pointcept` nuScenes checkpoint is supervised | research use | card verified; transfer to *detection* not shown | Utonia + CenterPoint head experiment (card §8) |
| LiDAR-only 3D detector | `centerpoint-nuscenes` (multi-class + velocity); `pointpillars-openpcdet` (simplest deploy) | nuScenes- or KITTI-like sensor | OpenPCDet zoo via secondary source | reproduce zoo number (Lab 05 B) |
| Camera-only 3D | `sparse4d-v3`, `streampetr` (sparse, temporal); `bevformer` (dense BEV) | 6-camera rig, accurate calibration | mostly not verified | licence check for StreamPETR; checkpoint links |
| Camera-LiDAR fusion | `bevfusion-mit` | calibrated, synced rig | Autoware/Qualcomm cards verified; MIT repo secondary | repo archived, so check maintenance |
| Easiest reproducible fine-tuning path | RT-DETRv2 (HF Trainer) for 2D; OpenPCDet PointPillars for 3D | — | — | — |
| Most deployment-ready | 2D: RT-DETRv2 → ONNX → TensorRT. 3D fusion: Autoware BEVFusion ONNX → TensorRT | TensorRT on NVIDIA | Autoware card verified | engine must be built locally |

## Decision procedure

1. **Constraints first:** licence, hardware (TOPS, memory), latency budget (p99, not mean), sensors available.
2. **Shortlist** with `model-lab registry list --task ...` and drop anything whose licence fails.
3. **Reproduce** one reported number for the top candidate before trusting the others.
4. **Evaluate on your data** with distance, size, weather and time-of-day stratification.
5. **Measure deployment** with the export → parity → engine → benchmark loop.
6. **Write a decision record** in `docs/decisions/` (template: [ADR 0001](decisions/0001-repository-structure-and-evidence-policy.md)).

## Trade-off axes

| axis | what to measure | trap |
|---|---|---|
| accuracy | AP / NDS by class, range, size | headline mAP hides far-range collapse |
| latency | p50/p99 end-to-end incl. pre/post on target HW | GPU timing without sync; batch-size games |
| label efficiency | metric vs % labels (1/10/100%) | different schedules per point |
| domain fit | train-city vs new-city gap | random frame split leaks scenes |
| weights availability | registry `artifacts.official_pretrained.status` | community re-uploads ≠ official |
| licensing | code, weights, dataset separately | permissive code with NC weights |
| deployment complexity | custom ops, plugins, out-of-graph steps | "exports to ONNX" ≠ "runs in TensorRT" |
