# Model registry

- `registry.yaml`: the curated entries (source of truth)
- `registry.schema.json`: generated from `src/model_lab/registry/schema.py` (`model-lab registry schema --write`). Editors can use it for YAML autocompletion.
- `cards/`: reference-quality model cards (12-section template, see any card)

## Field guide

| field | meaning |
|---|---|
| `role` | `foundation_encoder` · `task_detector` · `open_vocab_detector` · `segmenter` · `fusion_model` · `temporal_detector` · `tracker` |
| `pretraining.domain_fit` | `automotive` · `mixed_includes_automotive` · `generic_web_images` · `generic_detection_coco_like` · `indoor_or_object_3d` · `unknown` |
| `licensing.commercial_use` | `permissive` · `conditional` · `non_commercial` · `copyleft` · `unclear_legal_review` |
| `artifacts.<kind>.status` | `available_official` · `available_community` · `community_conversion` · `build_recipe_available` · `requires_local_build` · `gated` · `not_offered` · `not_found_after_search` · `not_verified` · `not_applicable` |
| `benchmarks[].claim_type` | `reported_paper` · `reported_repo` · `reproduced_local` |
| `verification.status` | `verified_primary` · `partially_verified` · `not_verified` |

## Card template (12 sections)

1. What it is (encoder vs detector vs foundation model) · 2. Best/poor fit · 3. Architecture
intuition · 4. I/O contract · 5. Pretraining data and domain match · 6. Links and licence ·
7. Inference · 8. Fine-tuning (frozen vs trainable) · 9. Export / quantisation / engine status ·
10. Benchmarks (published vs local plan) · 11. Failure modes and alternatives · 12. Sources and
last-verified date.
