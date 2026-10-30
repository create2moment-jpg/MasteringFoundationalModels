# DINOv2 ViT-B/14: image encoder for driving cameras

> Registry id: `dinov2-vitb14` · Last verified: 2026-10-30 · Status: partially verified
> (licence, parameter count and task come from HF metadata; the rest is from prior knowledge, see §12)

## 1. What it is

A **pretrained foundation encoder**. It is not a detector. It maps an image to a CLS
embedding plus a grid of patch embeddings, and it has no task head and no label set.
To detect anything you must attach and train a head (linear probe, DETR-style decoder, or
FPN + detection head), which turns it into a *model initialised from pretrained weights*.

| Concept | DINOv2 is… |
|---|---|
| (a) pretrained foundation encoder | **yes** |
| (b) task-specific detector | no |
| (c) foundation model that directly performs detection | no |
| (d) initialisation for a detector | yes, its main use in perception |
| (e) multimodal / open-vocabulary | no (image only) |

## 2. Best-fit and poor-fit use cases

**Best fit:** frozen-feature baselines for 2D detection and segmentation on driving images;
dense features for monocular depth; a backbone for camera-only BEV models (replacing ResNet-50 /
V2-99); label-efficient adaptation (≤ a few thousand labelled frames); feature-based data mining
and near-duplicate removal.

**Poor fit:** hard real-time on embedded hardware at full camera resolution (ViT cost grows
quadratically with token count); small far-range objects at 224 px input; any setting where a
COCO-trained CNN detector already meets the requirement.

## 3. Architecture intuition

A standard ViT-B (12 blocks, 768-d, 12 heads), patch size 14. It is trained by
self-distillation (a DINO image-level loss plus an iBOT masked-patch loss) on 142M curated
images. The student matches an EMA teacher across crops, so features become invariant to
augmentation while patch tokens stay spatially discriminative. **Why it transfers:** no labels
means no bias toward the 1000 ImageNet classes, and the masked-patch objective keeps local
detail, which dense tasks need.

## 4. Input / output contract

| | |
|---|---|
| Input | `float32 (B, 3, H, W)`, RGB, scaled to [0,1] then ImageNet mean/std normalised |
| Constraint | `H % 14 == W % 14 == 0` |
| Output | `last_hidden_state (B, 1 + (H/14)(W/14), 768)`: token 0 is CLS |
| Dense map | `tokens[:, 1:].reshape(B, H/14, W/14, 768)` |
| Example | 1600x900 nuScenes frame → resize to 1036x588 (74x42 patches = 3108 tokens) |

`AutoImageProcessor` defaults to resize-256 / centre-crop-224. **For driving images, disable
the centre crop.** It removes the left and right edges of a wide-FOV image, which is where
cut-in vehicles appear.

## 5. Pretraining data and domain match

LVD-142M is curated web imagery. Compared with driving data it lacks: a fixed ego viewpoint
and horizon, night and rain, motion blur, HDR sensor artefacts, rolling-shutter effects,
fisheye surround cameras, and extremely small far objects. Expect a **domain gap**. Measure it
in Lab 02 (frozen-probe mAP on nuScenes/BDD crops against an ImageNet ResNet-50).

## 6. Official links and licence

- Code: https://github.com/facebookresearch/dinov2
- Weights: https://huggingface.co/facebook/dinov2-base (Apache-2.0 per HF licence tag)
- Your fine-tuning dataset licence (nuScenes, BDD100K, …) governs any derived weights.

## 7. Run inference

```python
# Needs: pip install torch transformers pillow   (not executed in the build container: no torch)
import torch
from transformers import AutoModel

model = AutoModel.from_pretrained(
    "facebook/dinov2-base", revision="main"
).eval()  # pin a commit hash in real work
x = torch.randn(1, 3, 588, 1036)  # replace with a normalised driving frame
with torch.inference_mode():
    tok = model(pixel_values=x).last_hidden_state  # (1, 1 + 42*74, 768)
fmap = tok[:, 1:].reshape(1, 42, 74, 768).permute(0, 3, 1, 2)  # (1, 768, 42, 74)
```

## 8. Fine-tuning: what is frozen, what is trained

The decision tree is in [docs/04_finetuning_handbook.md](../../docs/04_finetuning_handbook.md).

| Strategy | Trainable | Typical LR | When |
|---|---|---|---|
| Linear probe | head only | 1e-3 (AdamW) | sanity check / feature quality |
| Frozen encoder + DETR head | head (~10–40M) | 1e-4 | < 5k labelled images |
| LoRA (r=8–16 on qkv/proj) | adapters + head | 1e-4 / 5e-4 | GPU-memory bound |
| Partial unfreeze (last 4 blocks) | last blocks + head | 1e-5 enc / 1e-4 head | moderate data, domain gap |
| Full fine-tune + LLRD 0.65–0.8 | all | 5e-5 peak | > 20k images, big domain gap |

## 9. Export / quantisation / engine status

| Artifact | Status |
|---|---|
| Official checkpoint | available (HF) |
| Automotive fine-tune | none known |
| Quantised weights | not verified |
| ONNX | build locally (`torch.onnx.export`, opset ≥ 17, export at deployment resolution) |
| TensorRT engine | build locally from ONNX. Engines are tied to GPU + TRT version, so never download one |

## 10. Benchmarks

The registry has no numbers yet. Published DINOv2 results are on generic benchmarks (ImageNet
linear, ADE20k, NYU depth). **They do not stand in for automotive validation.** Local plan:
Lab 02 compares frozen DINOv2-B against an ImageNet R50 on 2D detection over nuImages or BDD100K
val, with distance and size stratification, under the same head, schedule and resolution.

## 11. Failure modes, debugging checklist and alternatives

- Features look "blocky" at object boundaries: patch 14 is coarse. Use a higher input resolution or a ViT-adapter / FPN.
- Night images collapse in feature space: check with a t-SNE or feature-similarity probe split by `day/night` metadata.
- Accuracy is fine on the val split but drops on a new city: geographic domain shift, so evaluate per city.
- Alternatives: `dinov3-vitb16` (gated, custom licence), `c-radiov3-b` (high-res, NVIDIA licence), `siglip2-base-p16-224` (language-aligned).

## 12. Sources

- HF model page (licence, params): https://huggingface.co/facebook/dinov2-base (accessed 2026-10-30)
- Paper: https://arxiv.org/abs/2304.07193 (not re-read this session)
