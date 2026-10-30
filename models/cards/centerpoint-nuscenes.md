# CenterPoint: outdoor LiDAR 3D detector

> Registry id: `centerpoint-nuscenes` · Last verified: 2026-10-30 · Status: **not verified** (prior knowledge; GitHub unreachable from the build container)

## 1. What it is

A **task-specific 3D detector** (concept b) for LiDAR. Its backbone is VoxelNet (sparse conv)
or PointPillars. It has no foundation-model component.

## 2. Best-fit and poor-fit use cases

**Best fit:** multi-class nuScenes/Waymo-style detection with velocity, a simple tracking
baseline, and backbone swaps (it is the standard head for testing new 3D encoders).
**Poor fit:** dense urban scenes with heavily overlapping objects of the same class (the
centre-heatmap Gaussian radius becomes the limit), and very small objects at long range with a
coarse BEV stride.

## 3. Architecture intuition

Points → voxelise (0.075 or 0.1 m on nuScenes) → 3D sparse-conv backbone → flatten Z → 2D BEV
neck → **per-class-group heatmap of object centres** plus regression maps (sub-voxel offset,
height, log-size, sin/cos yaw, velocity). Objects are points, so rotation is just a regressed
attribute. That avoids the axis-aligned anchor mismatch of anchor-based heads. Tracking: a
detection at *t* projected back by `-v·Δt` is greedily matched to the closest track.

## 4. Input / output contract

| | |
|---|---|
| Input | `(N, 5)`: `x, y, z, intensity, Δt` from 10 motion-compensated sweeps (nuScenes) |
| Frame | LiDAR frame, x forward, y left, z up (same as `model_lab.data.geometry`) |
| Range | e.g. `[-54, -54, -5, 54, 54, 3]` m for the 0.075 voxel config |
| Output | per box: `x, y, z, l, w, h, yaw, vx, vy`, score, label. The z centre convention depends on the framework (mmdet3d stores the bottom centre) |
| Post-process | top-k on heatmap, decode, then circle-NMS (small classes) or rotated NMS |

## 5. Pretraining data and domain match

Trained on the target domain (nuScenes / Waymo). Moving to a different LiDAR (beam count,
mounting height, FOV) still causes large drops. Typical fixes: re-fit the z-range and ground
height, resample beams, and run domain adaptation (Stage 5 of the curriculum).

## 6. Links and licence

- Original: https://github.com/tianweiy/CenterPoint (MIT, prior knowledge)
- Also in OpenPCDet and mmdetection3d (Apache-2.0)
- nuScenes-trained weights fall under nuScenes terms (non-commercial). Waymo-trained weights are generally not redistributed.

## 7. Run inference

Recipe: [`recipes/lidar_3d_detection/README.md`](../../recipes/lidar_3d_detection/README.md)
(OpenPCDet route). The data-free intuition lab is
[`labs/05_lidar_3d_baseline`](../../labs/05_lidar_3d_baseline/README.md).

## 8. Fine-tuning

Freeze nothing by default: 3D detectors are trained from scratch or fine-tuned fully. For a new
class set, re-initialise the heatmap heads. For a new sensor, retrain with the same backbone
config first, then adjust voxel size or range one variable at a time.

## 9. Export / engine

The sparse-conv backbone needs a vendor library or plugins. NVIDIA publishes a CUDA/TensorRT
CenterPoint in Lidar_AI_Solution (prior knowledge, **verify**). The deliverable is a **build
recipe**, not a downloadable engine. Voxelisation, sweep accumulation and NMS stay outside the
network.

## 10. Benchmarks

Not transcribed (the paper was not re-read this session). When filled in, record the split
(val vs test), the number of sweeps, test-time augmentation (CenterPoint leaderboard entries
often use TTA + ensembles), and the metric (mAP and NDS).

## 11. Failure modes

- Velocity is wrong but boxes are right: the Δt channel is missing or sweeps are not motion-compensated.
- Boxes float or sink by h/2: bottom-centre vs gravity-centre mix-up (`bottom_to_gravity_center`).
- Pedestrians are merged: the heatmap Gaussian radius is too large, or the BEV stride is too coarse.

## 12. Sources

- https://arxiv.org/abs/2006.11275 (paper). Verification queue: repo licence, checkpoint URLs, CUDA-CenterPoint path.
