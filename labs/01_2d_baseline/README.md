# Lab 01: 2D detection baseline on driving images

**Objective:** run a closed-set detector (RT-DETRv2-R50, COCO weights) on driving frames, map
COCO classes to a driving taxonomy, understand the I/O contract, and evaluate it correctly.

**Status in this repo:** `infer_hf.py` is **written but not executed** (the build container had
no torch, no GPU and no Hub downloads). Part B runs on CPU.

## Prerequisites

- Stage 0 docs: [`docs/06_data_geometry_and_evaluation.md`](../../docs/06_data_geometry_and_evaluation.md) §2D metrics
- A few driving frames you are allowed to use, for example nuImages samples (non-commercial terms) or your own dash-cam images
- `torch` installed for your platform, then `pip install -e ".[vision]"`

## Part A: inference (GPU optional; CPU works but is slow)

```bash
python labs/01_2d_baseline/infer_hf.py --images data/frames/*.jpg --out outputs/lab01 --revision main
```

Expected: one JSON per image with `label`, `coco_label`, `score`, `box_xyxy`, and timings split
into preprocess, forward and postprocess. The script **fails loudly** if the class-map IDs do not
match `model.config.id2label`, which guards against the COCO 80-vs-91 id trap.

Pin `--revision` to a commit SHA from the HF repo's history and record it.

## Part B: evaluation on CPU (no model needed)

The metric code is in `src/model_lab/evaluation/ap.py`. Read it, then answer:

1. Why is RT-DETR post-processing a *top-k over (query, class) pairs* rather than NMS? What happens to AP if you add NMS anyway?
2. With `Criterion.IOU_2D`, compute AP at IoU 0.5 and at 0.75 on `tests/test_ap.py`-style hand-made frames. Which one penalises a 2-pixel box offset on a 20-pixel-tall pedestrian?
3. COCO "AP_small" uses area < 32². In a 1600x900 nuScenes frame, roughly how far away is a 1.7 m pedestrian at that size (focal length ≈ 1260 px)?

## Why it works / when it fails

- **Works:** cars, buses and trucks in daylight at moderate range, because these classes are COCO-rich.
- **Fails:** cones and barriers (absent from the taxonomy); night glare; heavy truncation at image borders (centre-crop pre-processing makes this worse); far pedestrians at 640 px input.

## Self-check quiz

1. What is the shape of `pred_boxes`, and in which coordinate convention?
2. Why must you report per-class AP only on mapped classes?
3. A 1600x900 image is resized to 640x640. What happens to box aspect ratios, and how do you undo it?

**Stretch:** fine-tune on nuImages or BDD100K with the 9-class driving map (`recipes/2d_detection/README.md`) and compare AP_small before and after.

## Cleanup

`rm -rf outputs/lab01`. HF weights are cached under `~/.cache/huggingface`.
