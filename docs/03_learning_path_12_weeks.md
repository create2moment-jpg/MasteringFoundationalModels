# 03: Learning path (stages 0–7, 12 weeks)

Assumes about 8–12 h/week, strong Python, PyTorch and engineering skills. GPU-heavy items are
marked 🖥 and each has a CPU or reduced-scale alternative. Depth beats breadth: one reproduced
number is worth ten skimmed papers.

## Stages

| stage | focus | labs | exit criterion |
|---|---|---|---|
| 0 Foundations | taxonomy, tensors, frames, benchmark literacy | 00, 04 | explain every box convention in `geometry.py`; pass Lab 04 quiz |
| 1 2D detection | backbone/neck/head, losses, decoding, eval | 01 | run RT-DETRv2 + class map; compute AP by hand on a toy set |
| 2 Perception encoders | frozen vs fine-tuned features, domain transfer | 02 (recipe) | strategy ranking table on a driving subset |
| 3 LiDAR / 3D | voxels, sparse conv, 3D boxes, pretrained point encoder | 04, 05, 06 | reproduce a PointPillars zoo number within about 1 AP |
| 4 Camera 3D + fusion | calibration, BEV transforms, fusion failure analysis | 07 | fusion vs LiDAR-only by range, with calibration perturbation |
| 5 Adaptation mastery | partial/full FT, PEFT, distillation, pseudo-labels | 08 | controlled ablation with 3 seeds |
| 6 Efficient inference | export, parity, PTQ/QAT, engines | 09 | engine with a parity report and benchmark record |
| 7 Research practice | reproduce, ablate, failure taxonomy, ADR | 10 | written decision record for one model choice |

## 12-week plan

| wk | goals | deliverables | time | checkpoint |
|---|---|---|---|---|
| 1 | env, registry, geometry | Lab 00 + 04 run; quiz answers in `notes/` | 6–8 h | can convert KITTI → lab boxes by hand |
| 2 | 2D detection anatomy | Lab 01 A+B; read RT-DETR paper §3 | 8–10 h | explain why there is no NMS |
| 3 | 2D fine-tune 🖥 (CPU alt: 500-image subset at 320 px) | nuImages/BDD fine-tune; per-class AP_S table | 10–12 h | AP_S improves over COCO-mapped baseline |
| 4 | image encoders 🖥 | `finetune_encoder.py` linear_probe vs frozen_head vs lora | 10 h | strategy ranking with seeds |
| 5 | LiDAR fundamentals | Lab 05 A; L-shape fitting stretch | 8 h | explain IoU-vs-distance AP gap |
| 6 | LiDAR detector 🖥 | OpenPCDet PointPillars KITTI reproduction | 10–12 h | within about 1 AP of zoo |
| 7 | point encoders 🖥 (CPU alt: feature probe on 20 nuScenes-mini sweeps) | Utonia/PTv3 feature extraction + linear seg probe | 10 h | domain-fit note added to the card |
| 8 | camera 3D & calibration | BEV projection exercises; StreamPETR or BEVFormer inference 🖥 | 10 h | calibration-perturbation experiment |
| 9 | fusion 🖥 | BEVFusion inference; per-range fusion vs LiDAR-only | 10–12 h | failure taxonomy draft |
| 10 | adaptation | distillation or pseudo-labelling experiment | 10 h | ablation table, 3 seeds |
| 11 | deployment 🖥 (CPU alt: Lab 09 A + ORT) | RT-DETRv2 → ONNX → TRT FP16/INT8 + parity | 10 h | parity report meets acceptance criteria |
| 12 | research practice | reproduce one published automotive result; ADR | 10 h | decision record merged |

## Self-assessment per stage

Each lab README ends with a quiz and a stretch exercise. Keep answers in a private `notes/`
folder (git-ignored). Re-take the Stage 0 quiz at week 12. If any answer changed, write down why.
