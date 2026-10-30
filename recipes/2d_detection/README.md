# Recipe: fine-tuning a 2D detector for driving classes

Model: RT-DETRv2-R50 ([card](../../models/cards/rt-detrv2-r50vd.md)). **Not executed in this repo.**

## Objective and when to use it

Adapt a COCO detector to a driving taxonomy (cones, barriers, riders, and so on) on nuImages or
BDD100K. Use it when COCO classes are insufficient, or when AP_small on your cameras is too low.

## Two paths

| path | how | when |
|---|---|---|
| HF Trainer | `RTDetrV2ForObjectDetection.from_pretrained(..., num_labels=K, id2label=..., ignore_mismatched_sizes=True)` + COCO-format dataset + `Trainer` | quick experiments, small data |
| Official repo | `lyuwenyu/RT-DETR` configs, multi-GPU, EMA, official schedules | reproducing paper numbers, long schedules |

## Parameter-freezing plan

1. **Warm-up head only** (2–5 epochs): freeze backbone and encoder, train the decoder and class head with the new label set.
2. **Full fine-tune**: unfreeze everything. Backbone LR is 0.1× the base LR (1e-4 AdamW, wd 1e-4), with EMA 0.9999, gradient clipping 0.1, and 36–72 epochs.

## Driving-specific configuration

- Input: keep aspect ratio, for example 1280x736 (divisible by 32), instead of 640x640.
- Augmentation: horizontal flip is OK. **No vertical flip** (the horizon prior is real). Photometric and night augmentation is useful. Mosaic is fine early but should be disabled in the last epochs.
- Class imbalance: repeat-factor sampling for rare classes (cones, construction vehicles).

## Evaluation

pycocotools on COCO-format GT: AP, AP50, AP75, AP_S/M/L, and per class. Split by
`weather` / `timeofday` attributes (BDD100K) to find weaknesses.

## Checklist

- [ ] `id2label` matches dataset category IDs (0- vs 1-based!)
- [ ] Val split is scene- or sequence-disjoint
- [ ] Baseline (COCO weights, mapped classes) evaluated before fine-tuning
- [ ] Report AP_small, not just AP
