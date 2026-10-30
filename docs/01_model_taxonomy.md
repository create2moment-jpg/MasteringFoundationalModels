# 01: Model taxonomy for autonomous-driving perception

The single most common mistake in this field is to call everything a "foundation model".
This repo uses five distinct concepts, and every registry entry and model card says which one applies.

| | Concept | Has a task head? | Fixed label set? | Example in registry |
|---|---|---|---|---|
| (a) | **Pretrained foundation encoder** | no | n/a | `dinov2-vitb14`, `c-radiov3-b`, `utonia`, `sonata` |
| (b) | **Task-specific detector / segmenter** | yes | yes | `rt-detrv2-r50vd`, `pointpillars-openpcdet`, `centerpoint-nuscenes`, `bevfusion-mit` |
| (c) | **Foundation model that directly performs a task** | yes (general) | no / promptable | `sam2.1-hiera-large` (promptable masks) |
| (d) | **Model initialised from pretrained weights** | yes | yes | any (b) whose backbone started from (a) |
| (e) | **Multimodal / open-vocabulary model** | yes | no (text-defined) | `grounding-dino-base`, `owlv2-base-p16-ensemble` |

The registry `role` enum is the operational version of this table: `foundation_encoder`,
`task_detector`, `open_vocab_detector`, `segmenter`, `fusion_model`, `temporal_detector`, `tracker`.

## The driving-perception stack, layer by layer

```
sensors ──► representation ──► encoder/backbone ──► view transform / fusion ──► temporal ──► head ──► post-proc ──► tracks
camera      image tensor        CNN / ViT (a)        2D→BEV (LSS, BEVFormer,     memory      2D/3D     NMS / top-k
LiDAR       points/voxels/      PointNet / sparse    PETR queries), BEV concat,  queues,     boxes,    decoding
            pillars/range img   conv / PTv3 (a)      point painting              BEV warp    seg       
```

### Camera 2D detection

- **Anchor-based CNN** (Faster R-CNN, RetinaNet, older YOLO): dense predictions relative to anchor boxes, then NMS.
- **Anchor-free CNN** (FCOS, CenterNet, YOLOX/YOLOv8+/YOLO11): predict per-location boxes or centres, then NMS.
- **Query-based / DETR family** (DETR, Deformable DETR, DINO-DETR, RT-DETR): a fixed set of learned queries with Hungarian matching. **No NMS.**
- **Open-vocabulary** (Grounding DINO, OWLv2): class scores come from similarity to text embeddings, so the label set is chosen at inference time.

### Image encoders used by driving stacks

- Supervised ImageNet CNNs (ResNet-50/101, V2-99 in many BEV papers) are the historical default.
- Self-supervised ViTs (DINOv2/v3), agglomerative ViTs (RADIO), and language-aligned encoders (CLIP / SigLIP) are newer. They give stronger frozen features but cost more compute per pixel.
- **Transfer questions for driving:** fixed horizon, wide aspect ratio, small and far objects, night and weather, multi-camera consistency, and fisheye distortion.

### LiDAR representations

| representation | encoder | pros | cons | examples |
|---|---|---|---|---|
| raw points | PointNet++ / point transformer | exact geometry | neighbour search cost | PointRCNN, PTv3 |
| voxels | 3D sparse conv | mature, accurate | sparse-conv deployment | SECOND, CenterPoint, PV-RCNN |
| pillars | PointNet per column → 2D CNN | dense 2D, easy deploy | loses vertical detail | PointPillars |
| range image | 2D CNN on (θ, φ) grid | dense, fast | scale varies with range, boundary blur | RangeDet, LaserNet |
| sparse transformer | windowed attention on voxels | long-range context | custom kernels | DSVT, VoxelNeXt (fully sparse head) |

### Point-cloud foundation encoders

Self-supervised point encoders (Sonata, Concerto, Utonia, all on a PTv3 backbone) are a
young area. Most pretraining mixtures are **indoor-heavy**. Utonia and the Concerto outdoor
checkpoint explicitly include outdoor LiDAR. **No automotive detection head ships with any of
them** at the baseline date, so treat them as transfer-learning candidates and measure the
benefit (see `models/cards/utonia.md` §8).

### Camera-only 3D

- **Depth-based lifting** (LSS, BEVDet, BEVDepth) predicts per-pixel depth distributions and splats features into BEV.
- **Query-based BEV** (BEVFormer) uses dense BEV queries that attend to image features via projected reference points.
- **Sparse queries** (DETR3D, PETR/StreamPETR, Sparse4D) use object queries with 3D reference points or 3D positional encodings. They are temporal through query propagation.

### Camera-LiDAR fusion levels

| level | how | example | failure sensitivity |
|---|---|---|---|
| point (early) | paint points with image semantics | PointPainting | calibration, time sync |
| feature / BEV | fuse BEV feature maps | BEVFusion | calibration (camera BEV shift) |
| proposal | LiDAR proposals query image features | TransFusion | missing LiDAR returns |
| late | merge per-sensor detections | classic tracking stacks | duplicate handling |

## What "foundation model for driving" can honestly mean today

1. A generic encoder (a) used as a backbone and **validated on driving data**.
2. A driving-pretrained encoder (self-supervised on fleet data). It is mostly proprietary, with few public checkpoints.
3. An open-vocabulary model (e) used **offline** for auto-labelling and rare-class mining.

Public evidence for (1) on 3D detection benchmarks is thinner than for 2D. The registry's
`pretraining.domain_fit` and `caveats` fields exist to make that visible.
