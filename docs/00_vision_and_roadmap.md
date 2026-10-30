# 00: Vision and roadmap

## Vision

A living, evidence-backed reference for **autonomous-driving perception** with foundation and
pretrained models: 2D camera detection, image encoders, LiDAR 3D detection, point-cloud
encoders, camera-only 3D and temporal perception, and camera-LiDAR fusion, plus fine-tuning,
evaluation, quantisation and deployment.

**Out of scope as standalone topics:** general LLM education, generic computer-vision
applications, indoor robotics and manipulation, planning, control, and motion forecasting.
Generic or indoor models appear only when their role as an encoder or adaptation starting
point for driving perception is explicit, and they are labelled with their domain fit.

## How to navigate

| question | go to |
|---|---|
| "Is model X an encoder or a detector? Can I ship it?" | `model-lab registry show <id>`, [`models/cards/`](../models/cards/) |
| "Which model should I use for Y?" | [02 selection playbook](02_model_selection_playbook.md) |
| "How do I fine-tune it?" | [04 handbook](04_finetuning_handbook.md), [`recipes/`](../recipes/) |
| "How do I deploy it?" | [05 deployment](05_quantization_and_deployment.md) |
| "Why don't my numbers match the paper?" | [06 evaluation](06_data_geometry_and_evaluation.md), [07 reproducibility](07_reproducibility_and_experiment_tracking.md) |
| "Where do I start learning?" | [03 learning path](03_learning_path_12_weeks.md) |

## Installation

```bash
# core (registry, geometry, metrics, CPU labs) – any OS, Python ≥ 3.10
uv venv && uv pip install -e ".[dev]"            # or: python -m venv .venv && pip install -e ".[dev]"
uv pip install -e ".[export]"                    # onnx + onnxruntime for Lab 09 A

# GPU stack – install torch FIRST for your CUDA version (see pytorch.org "get started"),
# then the HF extras:
uv pip install -e ".[vision]"
# LiDAR frameworks (OpenPCDet / mmdetection3d / spconv) are installed per their own docs
# at a pinned commit; they are not dependencies of this package.
```

## Roadmap

- **v0.1 (this version):** registry (23 entries) + schema + CLI; geometry, voxelisation, IoU/NMS/AP, benchmark harness; 5 model cards; executed CPU labs 04/05/09A; recipes for 2D, encoders, LiDAR and deployment; docs 00–09.
- **v0.2:** clear the verification queue (GitHub-hosted licences and checkpoints); Lab 02 encoder-transfer results on nuImages; Lab 06 Utonia/PTv3 feature probe on nuScenes-mini.
- **v0.3:** PointPillars KITTI reproduction record; RT-DETRv2 TensorRT parity report; camera-3D card (StreamPETR or Sparse4D).
- **v0.4:** fusion lab (07) with calibration-perturbation study; distillation lab (08); failure taxonomy (10).

See the final execution report in the project README for current blockers.
