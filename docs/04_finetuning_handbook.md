# 04: Fine-tuning and adaptation handbook

## Decision tree: what to try first

```
labels available on target domain?
├─ none ─────────► open-vocab auto-labelling (Grounding DINO / OWLv2 + SAM 2) → human QA → (8) pseudo-label loop
├─ < ~2k frames ─► (1) linear probe → (2) frozen encoder + head → (6) PEFT if supported
├─ 2k–20k ───────► (2) then (3) partial unfreeze with discriminative LR; (8) domain adaptation if city/sensor shift
└─ > 20k ────────► (4) full fine-tune with LLRD, EMA; (7) distil to deployment model
GPU memory tight? → (6) LoRA / (3) fewer unfrozen blocks / grad checkpointing / smaller input
deploy target fixed? → pick the student architecture now; plan (7) distillation and (11) QAT/PTQ early
```

## The 12 topics: what, when, pitfalls

| # | method | use when | pitfalls |
|---|---|---|---|
| 1 | Frozen features + linear probe | measuring representation quality | probe on CLS only discards spatial info; use dense probes for detection |
| 2 | New head on frozen encoder | small data, fast iteration | head capacity confounds encoder comparison; fix head across encoders |
| 3 | Partial / selective unfreezing | moderate data, domain gap | BatchNorm stats on small batches: freeze BN (`eval()`) in CNN backbones |
| 4 | Full fine-tuning | large data, large gap | catastrophic forgetting at high LR; use LLRD and warm-up |
| 5 | Optimisation hygiene: discriminative LR, LLRD (0.65–0.85 for ViT), wd excluded from norms/bias, warm-up, cosine, EMA, grad clip, accumulation, bf16, checkpoint/resume | always | accumulation changes effective batch, so rescale LR; resume must restore optimizer, scheduler, EMA and RNG |
| 6 | PEFT: LoRA/DoRA, adapters, bias-only (BitFit), prompt tuning | memory-bound, many tasks/domains | **supported** for ViT attention via `peft`. **Experimental** for DETR decoders and **unsupported** for sparse-conv LiDAR backbones in mainstream tooling. Can underperform full FT on large domain gaps |
| 7 | Knowledge distillation (logit, feature, relation) | deploy small, train big | teacher must be frozen and in eval mode; feature KD needs aligned shapes (adapters); evaluate the student on the same protocol |
| 8 | Domain adaptation: SSL continued pretraining, pseudo-labels, teacher–student (mean teacher), normalisation re-estimation | new city, sensor, weather | confirmation bias in pseudo-labels: use high thresholds, class-balanced selection, and a held-out human-labelled set |
| 9 | Low-label: active learning, hard-example mining, class rebalancing | long-tail classes | uncertainty sampling picks outliers; mix with diversity sampling |
| 10 | Synthetic data / auto-labelling | rare scenarios | sim-to-real gap; never put synthetic frames in val/test; audit label noise |
| 11 | QAT vs PTQ | deployment accuracy loss | PTQ first; QAT only if PTQ loses > budget; calibration from deploy domain |
| 12 | Reproducibility: seeds, dataset versioning, config snapshots, tracking, lineage | always | see [07](07_reproducibility_and_experiment_tracking.md) |

> **Terminology warning:** LLM weight-only quantisation methods (GPTQ, AWQ) and LoRA tooling
> built for LLMs do not automatically apply to CNN detectors, sparse-conv LiDAR backbones or
> deformable-attention decoders. Verify per architecture.

## Per-recipe template (every recipe in `recipes/` follows this)

1. Objective and when to use it
2. Exact parameter-freezing plan
3. Input/output contract
4. Config and command-line examples
5. GPU/memory considerations and fallbacks
6. Losses and metrics
7. Checkpointing, resume, early stopping, best-checkpoint selection (by val metric, never test)
8. Validation protocol and common failure cases
9. Expected artifacts and interpretation checklist

Implemented: [`recipes/image_encoders`](../recipes/image_encoders/README.md) (strategies 1–5 plus LoRA),
[`recipes/2d_detection`](../recipes/2d_detection/README.md), [`recipes/lidar_3d_detection`](../recipes/lidar_3d_detection/README.md).

## False gains: check before celebrating

- Val frames from the same drive sequences as train frames (temporal leakage)
- Different eval resolution, score threshold or class map between runs
- Best epoch chosen on the test split
- Single seed with a delta smaller than seed variance
- A "frozen" baseline that accidentally updated BN statistics
