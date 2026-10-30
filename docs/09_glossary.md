# 09: Glossary

| term | meaning |
|---|---|
| AP / mAP | area under the precision–recall curve; mean over classes (and thresholds for COCO) |
| AP_R40 | KITTI AP sampled at 40 recall points (1/40…1) |
| APH | Waymo AP with heading-accuracy weighting |
| BEV | bird's-eye view: top-down grid in the ego/LiDAR frame |
| BEV pooling | summing camera-frustum features into BEV cells (LSS / BEVFusion) |
| calibration (sensor) | intrinsics (K, distortion) + extrinsics (sensor→ego transforms) |
| calibration (quant.) | collecting activation ranges for static INT8 quantisation |
| CBGS | class-balanced grouping and sampling (nuScenes long-tail trick) |
| circle NMS | suppression by BEV centre distance (CenterPoint) |
| domain fit | how well pretraining data matches the deployment sensors/scenes |
| engine / plan | compiled, hardware-specific inference artifact (TensorRT) |
| EMA | exponential moving average of weights; often the evaluated model |
| foundation encoder | large pretrained feature extractor without a task head |
| gravity centre | geometric 3D centre of a box (vs bottom centre) |
| Hungarian matching | one-to-one assignment between predictions and GT (DETR training) |
| LLRD | layer-wise learning-rate decay: smaller LRs for earlier layers |
| LoRA / DoRA | low-rank (and weight-decomposed) adapters for parameter-efficient fine-tuning |
| LSS | Lift-Splat-Shoot: per-pixel depth distribution lifting to BEV |
| NDS | nuScenes detection score: mAP plus true-positive error terms |
| NMS | non-maximum suppression |
| occupancy | 3D voxel grid of free/occupied (+ semantic) state |
| opset | ONNX operator-set version |
| parity check | comparing framework vs exported/quantised outputs on identical inputs |
| pillar | vertical column voxel spanning the full z-range |
| PTQ / QAT | post-training quantisation / quantisation-aware training |
| QDQ | Quantize/DequantizeLinear node pairs marking INT8 regions in ONNX |
| range image | LiDAR returns arranged on a (beam, azimuth) grid |
| serialisation (PTv3) | ordering points along space-filling curves to form attention patches |
| sparse convolution | convolution computed only at non-empty voxels (spconv, MinkowskiEngine) |
| sweep | one LiDAR revolution; multi-sweep = several sweeps aggregated with Δt |
| TTA | test-time augmentation; inflates leaderboard numbers |
| voxel | 3D grid cell; voxelisation = assigning points to cells |
