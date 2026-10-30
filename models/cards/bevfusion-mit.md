# BEVFusion (MIT HAN Lab): camera-LiDAR fusion, including the deployment path

> Registry id: `bevfusion-mit` · Last verified: 2026-10-30 · Status: partially verified
> (Autoware and Qualcomm model cards read; MIT repo and NVIDIA path via secondary sources)

## 1. What it is

A **fusion model with a task head** (concept b, multi-sensor). It is not a foundation model.
Its image backbone is ImageNet- or nuImages-pretrained, and its LiDAR backbone is a
sparse-conv VoxelNet.

## 2. Best-fit and poor-fit use cases

**Best fit:** surround-camera plus LiDAR 3D detection and BEV map segmentation with a mature
TensorRT path (NVIDIA CUDA-BEVFusion, Autoware).
**Poor fit:** setups without accurate, time-synchronised calibration; camera rigs very different
from nuScenes without retraining.

## 3. Architecture and the full dependency chain

```
6 cameras ─► image backbone + FPN ─► depth distribution per pixel (LSS) ─► BEV pooling ─┐
                                    (needs K, cam2ego, image-aug matrices)              ├─► concat ─► BEV conv encoder ─► TransFusion head ─► boxes
LiDAR sweeps ─► voxelise ─► sparse-conv backbone ─► flatten Z ─► LiDAR BEV ─────────────┘
              (needs lidar2ego, ego poses for sweep compensation)
```

Fusion happens at the **BEV feature level**. The camera branch carries semantic density and the
LiDAR branch carries geometry. It is robust to one weak modality, but not to *wrong
calibration*: a 0.5° yaw error in a camera extrinsic shifts camera BEV features by about 0.9 m at 100 m.

## 4. Input / output contract

| | |
|---|---|
| Images | 6 x `(3, 256, 704)` after resize/crop (nuScenes config) |
| Calibration | `camera_intrinsics`, `camera2lidar`, `lidar2image`, `img_aug_matrix`, `lidar_aug_matrix` |
| LiDAR | 9–10 sweeps `(x, y, z, intensity, Δt)` |
| Output | 3D boxes + velocity (10 nuScenes classes), and/or BEV segmentation maps |

## 5. Pretraining data and domain match

Trained on nuScenes (Boston and Singapore, 32-beam LiDAR, 6 cameras). The Autoware variants
are a **different model instance**: 5 classes, roughly 120 m range, their own training data and
camera ROI 576x384 from 1440x1080. Do not mix their metrics with paper metrics.

## 6. Links and licence

- Code: https://github.com/mit-han-lab/bevfusion (Apache-2.0 per secondary source; repo archived)
- Autoware ONNX: https://huggingface.co/AutowareFoundation/bevfusion (Apache-2.0, model card)
- NVIDIA deployment: https://github.com/NVIDIA-AI-IOT/Lidar_AI_Solution (CUDA-BEVFusion; licence reported as "other", **legal review**)
- nuScenes weights: nuScenes terms (non-commercial)

## 7. Run inference

The research route is the MIT repo (mmdet3d 0.x era, custom CUDA ops), so pin its documented
CUDA/PyTorch versions. The deployment route is described below.

## 8. Fine-tuning

The usual schedule is a LiDAR-only pretrain, then fusion fine-tuning with the camera branch
added. For a new rig, recompute every calibration matrix and keep image-augmentation matrices
consistent between training and inference. That last point is the most common silent bug.

## 9. Deployment path: artifacts kept distinct

| Layer | Artifact | Status |
|---|---|---|
| Training checkpoint | `.pth` (MIT repo) | not verified this session |
| Quantised weights | INT8 via NVIDIA CUDA-BEVFusion PTQ | reported (secondary); not verified |
| Portable graph | ONNX: Autoware `bevfusion_lidar.onnx`, `bevfusion_camera_lidar.onnx`, `bevfusion_image_backbone.onnx` | **available (community/Autoware)** |
| Compiled engine | TensorRT plan | **build locally**. Autoware builds engines from ONNX on first launch, so no universal engine file exists |
| Runtime + plugins | TensorRT + `spconv_cpp` / `autoware_tensorrt_plugins` (sparse conv), custom BEV pooling | required |
| Outside the graph | voxelisation, multi-frame densification, undistortion, circle/IoU NMS, yaw normalisation, score thresholds | implemented in the Autoware node |

**Parity check you must run:** feed identical sweeps and images to (a) PyTorch and (b)
ONNX/TensorRT. Compare the BEV feature map (cosine similarity > 0.99), the raw heatmap, and the
decoded boxes (IoU > 0.95 matched). Then compare mAP on a fixed 200-frame subset.

## 10. Benchmarks

Paper and leaderboard numbers are not transcribed here. Autoware and Qualcomm exports report
their own runtime figures on their own data and hardware, so record them only with that context.

## 11. Failure modes

- Camera-only objects appear shifted: an extrinsic or `img_aug_matrix` mismatch.
- Detection collapses when one camera drops: the model was trained without modality dropout, so add camera dropout in fine-tuning.
- Ghost boxes behind trucks: LiDAR occlusion combined with a camera depth smear. Inspect the BEV feature maps per branch.

## 12. Sources

- https://huggingface.co/AutowareFoundation/bevfusion (read 2026-10-30)
- https://huggingface.co/qualcomm/BEVFusion (read 2026-10-30; camera-BEV export for Snapdragon, a different scope)
- https://awesome.ecosyste.ms/projects/github.com%2Fmit-han-lab%2Fbevfusion (secondary)
- https://arxiv.org/abs/2205.13542
