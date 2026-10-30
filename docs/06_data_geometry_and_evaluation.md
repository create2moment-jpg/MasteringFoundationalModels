# 06: Data, geometry and evaluation foundations

Code references: `src/model_lab/data/geometry.py`, `voxelize.py`, `evaluation/{iou,nms,ap}.py`.

## 2D detection essentials

- **Box formats:** `xyxy` (pixels), `xywh` (COCO JSON), `cxcywh` normalised (DETR outputs). Mixing them is the top 2D bug.
- **Anchor-based vs anchor-free vs query-based:** see [01](01_model_taxonomy.md). Assignment (IoU-based, centre sampling, SimOTA, Hungarian) determines which predictions receive gradients.
- **Losses:** classification (CE / focal / varifocal), box regression (L1 + GIoU / CIoU), and quality/IoU-aware scores.
- **NMS** and **confidence calibration:** scores are rankings, not probabilities, until calibrated. Check reliability diagrams before thresholding for safety logic.

## 3D geometry

- **Frames:** sensor (LiDAR, each camera) → ego (vehicle) → global (world). nuScenes stores `sensor→ego` calibration per sensor and `ego→global` pose per timestamp.
- **Lab box convention:** `[x, y, z, l, w, h, yaw]`, gravity centre, yaw CCW from +x in the LiDAR frame (OpenPCDet). mmdetection3d uses bottom centre. KITTI labels are camera-frame bottom centre with `ry` about camera-y. nuScenes uses `wlh` order and quaternion orientation in the global frame.
- **Camera model:** `p_img ~ K [R|t] p_lidar` with intrinsics K (3x3) or projection P (3x4, KITTI rectified). Points with z ≤ 0 in the camera frame are behind the camera and must be masked.
- **Temporal alignment:** each LiDAR sweep has its own ego pose. Multi-sweep aggregation transforms past sweeps into the current frame and appends Δt. Camera and LiDAR timestamps differ by up to tens of ms, which at 20 m/s is 0.2–0.6 m.

## Dataset layouts and governance

| dataset | sensors | annotation | metric | licence note (verify) |
|---|---|---|---|---|
| KITTI | 1 LiDAR (64-beam), stereo | 3D boxes (front camera FOV) | AP_R40 by difficulty | CC BY-NC-SA 3.0 |
| nuScenes | 1 LiDAR (32-beam), 6 cams, 5 radars | 3D boxes, 10 det classes, attrs, velocity | mAP (centre distance) + NDS | non-commercial terms; commercial licence separate |
| Waymo Open | 5 LiDARs, 5 cams | 3D boxes, tracking | APH (L1/L2) | Waymo licence; derived weights restricted |
| Argoverse 2 | 2 LiDARs, 7 ring + 2 stereo cams | 26 classes | CDS / mAP | check terms |
| ONCE | LiDAR + 7 cams | 1M frames, few labelled | AP by distance | check terms |
| PandaSet | 2 LiDARs, 6 cams | boxes + seg | — | check terms |
| A2D2 | 5 LiDARs, 6 cams | boxes + seg | — | CC BY-ND 4.0 (verify) |
| BDD100K | dash cam | 2D boxes, seg, tracking, weather/time attrs | COCO-style AP | check terms |
| DAIR-V2X | vehicle + infrastructure | cooperative 3D | — | check terms |

Licences listed here come from prior knowledge, except where the registry evidence says
otherwise. See [08](08_licensing_and_data_governance.md) and verify each before use.

**Manifests:** every experiment should reference a versioned manifest (file list + hash +
split + class map) rather than "the nuScenes dataset".

## Metrics, and why they are not interchangeable

| metric | matching | notes |
|---|---|---|
| COCO mAP@[.5:.95] | 2D IoU, 10 thresholds | AP_S/M/L by pixel area; 101-point interpolation |
| KITTI 3D/BEV AP_R40 | 3D or BEV IoU (0.7 car, 0.5 ped/cyc) | difficulty levels by height, occlusion, truncation |
| nuScenes mAP | BEV centre distance {0.5, 1, 2, 4} m | AP clipped below 10% recall and precision |
| nuScenes NDS | 0.5·mAP + TP-error terms (ATE, ASE, AOE, AVE, AAE) | velocity and attribute errors matter |
| Waymo APH | 3D IoU, heading-weighted | LEVEL_1 / LEVEL_2 difficulty |
| AMOTA / AMOTP / ID switches | tracking only | do not use for detection-only models |

The executed Lab 05 A output shows one detector producing AP between 0.002 and 0.85
depending only on the matching criterion. Never compare numbers across protocols.

## Robustness evaluation

Stratify by range (0–20, 20–40, 40–80 m), object size, occlusion, weather, lighting, scene
density and rare classes. Add **sensor-fault tests**: drop a camera, add a 0.5° extrinsic
yaw error, add a 50 ms time offset, drop 50% of LiDAR beams. Report the delta against clean
data, not just absolute numbers.

## Benchmark harness

`model_lab.benchmarking.run_benchmark` records model ID and revision, dataset/split, config
hash, git commit, environment, input description, batch, precision, warm-up and iterations,
per-stage latency percentiles (pre-process / forward / post-process / end-to-end), throughput,
optional Python peak memory, task metrics, and checkpoint source. CPU smoke test:
`model-lab benchmark synthetic`.
