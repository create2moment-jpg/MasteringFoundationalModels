# Recipe: adapting image foundation encoders to driving cameras

Script: [`finetune_encoder.py`](finetune_encoder.py). **Not executed in this repo** (no torch in
the build container). It was syntax-checked and linted only.

## Objective

Rank adaptation strategies (linear probe → frozen dense head → LoRA → partial → full with
LLRD) for one encoder on one driving dataset, using **identical** data, resolution, schedule
and metric. Only then move to detector heads.

## Data contract

A JSONL manifest with `image`, `boxes_xyxy` and `split`. Build it from nuImages or BDD100K
vehicle boxes, and respect those datasets' terms. Use a **sequence- or city-disjoint** val split,
because frames from the same drive leak appearance.

## Parameter-freezing plan

| strategy | frozen | trainable | first LR | memory |
|---|---|---|---|---|
| linear_probe | encoder | 1x1 conv | 1e-3 | lowest (no encoder grads) |
| frozen_head | encoder | 3x3 conv head (~1.8M) | 1e-3 | low |
| lora | encoder base weights | LoRA on query/value (r=8) + head | head 1e-3, LoRA 2e-5…1e-4 | medium |
| partial | blocks 0..N-5 | last 4 blocks + head | 2e-5 encoder | medium-high |
| full | — | all, LLRD 0.75 | 2e-5 peak | highest |

## Commands

```bash
for s in linear_probe frozen_head lora partial full; do
  python recipes/image_encoders/finetune_encoder.py --manifest data/vehicles.jsonl \
      --strategy $s --epochs 10 --amp --out outputs/enc_$s --seed 0
done
```

Run 3 seeds per strategy if differences are under about 1 IoU point.

## GPU memory and fallbacks

At 518x924 (2442 tokens) ViT-B with full fine-tuning and batch 8 in bf16 needs on the order of
20+ GB (this is an estimate, not measured). Fallbacks: `--batch 2 --accum 4`, a lower resolution
(364x644), gradient checkpointing (`encoder.gradient_checkpointing_enable()`), or LoRA.

## Validation and interpretation checklist

- [ ] Same val split, resolution and threshold for all strategies
- [ ] Report patch IoU **and** recall on patches belonging to small boxes (< 32 px tall)
- [ ] If `full` < `partial`: suspect LR too high (catastrophic forgetting) or too few images
- [ ] If `linear_probe` ≈ `full`: the encoder features already suffice, so prefer the cheap path
- [ ] Log the train-val gap. LoRA and frozen strategies overfit less on small data

## Expected artifacts

`outputs/enc_<strategy>/best_head.pt` (head weights + args) and `log.json` (per-epoch metrics).
LoRA adapters are not saved by this script. Add `encoder.save_pretrained(...)` if you need them.
