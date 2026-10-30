# Lab 05: LiDAR 3D detection baseline

**Objective:** (A) understand 3D detection evaluation end to end with a classical baseline on
CPU; (B) reproduce a published learned baseline (PointPillars, KITTI) with OpenPCDet on a GPU.

## Part A: classical baseline + metric anatomy (CPU, executed 2026-10-30)

```bash
python labs/05_lidar_3d_baseline/run_cluster_baseline.py --scenes 20
```

Pipeline (`src/model_lab/baselines/clustering.py`): flat-ground removal → 8-connected BEV
clustering (0.3 m cells) → PCA-oriented box extended to the ground → score from point count.

Output as executed (20 synthetic scenes, 240 GT, 252 detections, numpy on 4 CPUs):

| criterion | all-class AP | car AP | 0–20 m | 20–40 m | 40–80 m |
|---|---|---|---|---|---|
| BEV IoU ≥ 0.7 | 0.002 | 0.004 | 0.007 | 0.000 | 0.000 |
| BEV IoU ≥ 0.5 | 0.094 | 0.138 | 0.250 | 0.078 | 0.000 |
| BEV IoU ≥ 0.25 | 0.262 | 0.276 | 0.633 | 0.244 | 0.002 |
| 3D IoU ≥ 0.5 | 0.072 | 0.109 | 0.241 | 0.015 | 0.000 |
| centre dist ≤ 1 m | 0.534 | 0.468 | 0.781 | 0.583 | 0.238 |
| centre dist ≤ 2 m | 0.702 | 0.493 | 0.887 | 0.711 | 0.477 |
| centre dist ≤ 4 m | 0.849 | 0.493 | 0.958 | 0.902 | 0.648 |

The same PR curve gives 0.702 (COCO-101), 0.682 (KITTI R40), 0.701 (VOC) and 0.657 (nuScenes-style).
Latency was p50 about 5 ms per sweep. **Synthetic, so it is not a detector result.**

**What this teaches**
1. LiDAR sees only the sensor-facing "L". Geometric fitting under-estimates size and shifts the centre, so strict IoU AP collapses while centre-distance AP looks decent. Learned detectors *regress the full box* from priors, and that is most of their value.
2. One curve gives four "AP" numbers. Always state the criterion, threshold, interpolation and range.
3. Car AP plateaus at 0.49 because a size rule (`l > 2.5 m`) misses cars seen end-on. Recall caps AP.
4. Far range (40–80 m) is where everything breaks. Always stratify.

## Part B: PointPillars on KITTI with OpenPCDet (GPU, **not executed here**)

Follow [`recipes/lidar_3d_detection/README.md`](../../recipes/lidar_3d_detection/README.md).
Success criterion: Car 3D AP_R40 (moderate) within about 1 point of the OpenPCDet model-zoo table
for the same config, evaluated with the *KITTI devkit protocol via OpenPCDet*, not this repo's
reference AP.

## Quiz

1. Why is nuScenes detection matched by centre distance and not IoU?
2. What does "R40" mean, and why did KITTI move away from R11?
3. Your detector's BEV IoU-0.7 AP is 10 points below its BEV IoU-0.5 AP. List three hypotheses and the test for each.

**Stretch:** replace PCA with an L-shape fitting search over yaw (minimise points-to-edge
distance) and measure the change in IoU-0.5 AP by range bin.
