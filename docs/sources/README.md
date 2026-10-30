# Source index and verification queue

Research baseline: **2026-10-30**. Access method per source: **HF** = Hugging Face Hub API or
model card read via connector; **WS** = web-search snippet (secondary); **PK** = prior
knowledge, not re-checked this session; **✗** = not reachable from the build container.

## Seed sources from the generating prompt

| source | status this session |
|---|---|
| https://ai.meta.com/research/dinov3/ | ✗ (proxy 403). DINOv3 facts from HF metadata (gated) |
| https://github.com/facebookresearch/dinov2 | ✗; HF `facebook/dinov2-base` checked (HF) |
| https://docs.nvidia.com/tao/tao-toolkit/latest/text/foundation_models/overview.html | ✗, not used |
| https://docs.nvidia.com/tao/tao-toolkit/latest/text/overview.html | ✗, not used |
| https://github.com/lyuwenyu/RT-DETR | ✗; HF `PekingU/rtdetr_v2_r50vd` checked (HF) |
| https://github.com/IDEA-Research/GroundingDINO | ✗; HF `IDEA-Research/grounding-dino-base` checked (HF) |
| https://github.com/Pointcept/Pointcept | ✗; HF `Pointcept/PointTransformerV3` card read (HF) |
| https://github.com/Pointcept/Utonia | ✗; HF `Pointcept/Utonia` card read (HF) |
| https://huggingface.co/Pointcept/Utonia | ✓ (HF) |
| https://github.com/open-mmlab/OpenPCDet | ✗; WS (mirrors) for licence/zoo; HF community re-uploads found |
| https://github.com/open-mmlab/mmdetection3d | ✗, not used |
| https://github.com/exiawsh/StreamPETR | ✗; repo URL corroborated via WS (Sparse4D README) |
| https://github.com/mit-han-lab/bevfusion | ✗; WS (licence, archived) + HF Autoware/Qualcomm cards |
| https://onnx.ai/models/ | ✗, not used |
| https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html | ✗; ORT quantisation API exercised locally (onnxruntime 1.31.1) |
| https://docs.nvidia.com/tensorrt/ | ✗, not used (PK only) |
| Torch-TensorRT quantisation docs | ✗, not used |
| TensorRT quantized types docs | ✗, not used (PK only) |
| OpenVINO supported models (2026) | ✗, not used |
| https://www.nuscenes.org/ | ✗ (PK for terms) |
| https://waymo.com/open/ | ✗ (PK) |
| https://www.argoverse.org/av2.html | ✗ (PK) |
| https://www.cvlibs.net/datasets/kitti/ | ✗ (PK) |
| https://huggingface.co/models | ✓ (HF search for BEVFusion, OpenPCDet ports, Pointcept) |

## Additional sources used

- https://huggingface.co/nvidia/C-RADIOv3-B: model card (HF)
- https://huggingface.co/Pointcept/Concerto, https://huggingface.co/Pointcept/Sonata, https://huggingface.co/facebook/sonata (HF)
- https://huggingface.co/google/siglip2-base-patch16-224, https://huggingface.co/google/owlv2-base-patch16-ensemble, https://huggingface.co/facebook/sam2.1-hiera-large (HF)
- https://huggingface.co/AutowareFoundation/bevfusion, https://huggingface.co/qualcomm/BEVFusion: model cards (HF)
- https://huggingface.co/torch-pointcloud: community OpenPCDet weight re-uploads (HF)
- https://awesome.ecosyste.ms/projects/github.com%2FHorizonRobotics%2FSparse4D (WS)
- https://awesome.ecosyste.ms/projects/github.com%2Fmit-han-lab%2Fbevfusion (WS)
- https://gitee.com/mayix/OpenPCDet (WS mirror)

## Verification queue (priority order)

1. OpenPCDet: licence file, model-zoo checkpoint URLs and filenames (PointPillars, SECOND, PV-RCNN, CenterPoint, VoxelNeXt, DSVT).
2. NVIDIA-AI-IOT/Lidar_AI_Solution: licence text (reported as "other"), CUDA-PointPillars, CUDA-CenterPoint, CUDA-BEVFusion paths and supported TensorRT versions.
3. StreamPETR, BEVFormer, Sparse4D: licence files and checkpoint links.
4. Sonata: authoritative licence (the two HF repos disagree or are incomplete) and pretraining-data composition.
5. DINOv3: licence text and register-token count (requires gated access).
6. RT-DETR official ONNX/TensorRT export path and its opset.
7. Ultralytics: current model generation, weight licence.
8. Dataset terms: nuScenes, Waymo, AV2, KITTI, BDD100K, A2D2 (current pages).

## Change process

See [CONTRIBUTING.md](../../CONTRIBUTING.md#updating-model-references): verify against a
primary source, update the entry and `verification` block, run `model-lab registry validate
--check-schema-file`, and note the change in the PR.
