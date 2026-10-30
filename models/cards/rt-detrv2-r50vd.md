# RT-DETRv2 R50vd: 2D detector baseline for driving cameras

> Registry id: `rt-detrv2-r50vd` · Last verified: 2026-10-30 · Status: partially verified

## 1. What it is

A **task-specific detector** (concept b). It is trained end-to-end on COCO's 80 classes and
outputs boxes directly. It is *initialised from* an ImageNet-pretrained ResNet-50vd. It is not
a foundation encoder and it is not open-vocabulary.

## 2. Best-fit and poor-fit use cases

**Best fit:** the default closed-set 2D baseline for driving cameras. It is permissively
licensed, NMS-free, and has a TensorRT-friendly path.
**Poor fit:** classes that COCO lacks (traffic cones, barriers, construction vehicles, and so on)
without fine-tuning; far-range pedestrians at 640 px input.

## 3. Architecture intuition

Backbone (R50vd, strides 8/16/32) → **hybrid encoder** (attention only on the stride-32 map,
then CNN cross-scale fusion, which is cheaper than full multi-scale attention) → **IoU-aware
query selection** that picks the top-300 encoder tokens as initial queries → a 6-layer
decoder with multi-scale deformable attention. Training uses Hungarian matching, so each object
gets exactly one prediction and **no NMS is needed**. v2 adds per-level sampling points
and a discrete-sampling option that is friendlier to deployment.

## 4. Input / output contract

| | |
|---|---|
| Input | `(B, 3, 640, 640)` RGB in [0,1]. Check `preprocessor_config.json` for normalisation |
| Output | `logits (B, 300, 80)` (sigmoid, not softmax), `pred_boxes (B, 300, 4)` normalised `cx, cy, w, h` |
| Post-process | sigmoid → top-k over (query, class) → scale boxes to original size. No NMS |

Driving frames are 16:9 or wider. Resizing 1600x900 to 640x640 squashes geometry. Prefer
fine-tuning at a native aspect ratio such as 1280x736, or letterbox and undo it exactly.

## 5. Pretraining data and domain match

COCO is generic. Its "car / truck / bus / person / bicycle / traffic light / stop sign" overlap
with driving taxonomies, but box statistics differ: driving data has more small, occluded and
truncated objects. Map classes explicitly (see `configs/class_maps/`) and **report per-class AP
only on classes that are mapped**.

## 6. Official links and licence

- Code: https://github.com/lyuwenyu/RT-DETR (Apache-2.0)
- Weights: https://huggingface.co/PekingU/rtdetr_v2_r50vd (Apache-2.0, 43.0M params per HF metadata)

## 7. Run inference

See [`labs/01_2d_baseline`](../../labs/01_2d_baseline/README.md). It contains a runnable script
(`infer_hf.py`, which needs torch + transformers) and a CPU-only metric walkthrough.

## 8. Fine-tuning

- **Head-only** (freeze backbone and encoder): quick adaptation to a new class map with few labels.
- **Full fine-tune**: the standard path. Backbone LR at 0.1× the base LR, AdamW 1e-4, EMA, 36–72 epochs on BDD100K/nuImages scale.
- PEFT: experimental for this architecture. Not recommended as the first try.
- Set `num_labels` and the id2label/label2id maps. In HF, use `ignore_mismatched_sizes=True` to re-initialise the class head.

## 9. Export / quantisation / engine status

| Artifact | Status |
|---|---|
| Official checkpoint | available (HF) |
| ONNX | official repo export scripts (not verified this session) |
| TensorRT | build locally (`trtexec --onnx=... --fp16`, then record TRT version, GPU and profile) |
| INT8 | needs calibration on **driving** images; check decoder AP drop per class |

The parity procedure is in [`labs/09_quantization_and_export`](../../labs/09_quantization_and_export/README.md).

## 10. Benchmarks

COCO val AP is reported in the paper. It is *not transcribed* here because the paper was not
re-read in this session. Local plan: fine-tune on BDD100K or nuImages and report AP, AP_S, and AP
by distance proxy (box height) with the harness in `src/model_lab/benchmarking`.

## 11. Failure modes and alternatives

- Many low-confidence duplicates after INT8: decoder quantisation error, so keep the decoder in FP16.
- Missing small objects: raise the input resolution or add a stride-8 query source.
- Alternatives: `yolo11-ultralytics` (AGPL), D-FINE / DEIM-family RT-DETR successors (not yet in the registry, so verify), and `grounding-dino-base` for open-set labelling.

## 12. Sources

- https://huggingface.co/PekingU/rtdetr_v2_r50vd (HF metadata, accessed 2026-10-30)
- https://arxiv.org/abs/2407.17140 (RT-DETRv2), https://arxiv.org/abs/2304.08069 (RT-DETR)
