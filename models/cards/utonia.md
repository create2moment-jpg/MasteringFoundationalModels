# Utonia: pretrained point-cloud encoder, with domain-fit analysis

> Registry id: `utonia` · Last verified: 2026-10-30 · Status: verified (HF model card)
> **Licence: CC-BY-NC-4.0, so research use only.**

## 1. What it is

A **self-supervised foundation encoder** for point clouds (concept a). It produces per-point
features, with no detection head and no label set. It is *not* a 3D detector. To detect cars
you need a detection adapter (see §8).

## 2. Best-fit and poor-fit use cases

**Best fit:** label-efficient LiDAR semantic segmentation; initialising the 3D backbone of a
detector when labels are scarce; cross-sensor transfer studies (a new LiDAR model or mounting
height); feature-space data mining.
**Poor fit:** commercial products (NC licence); latency-critical on-vehicle detection without
distillation; pipelines that depend on LiDAR intensity (not an input by default).

## 3. Architecture intuition

The backbone is Point Transformer V3. Points are grid-sampled, **serialised** along
space-filling curves (Z-order / Hilbert), and grouped into fixed-size patches. Attention runs
inside patches, and the serialisation order changes between blocks so the receptive field
mixes. Utonia trains *one* such encoder with self-supervision across very different domains:
remote sensing, outdoor LiDAR, indoor RGB-D, CAD, and video-lifted points. It does this by
making inputs uniform (zero-filling missing colour and normals) and sharing the representation
space.

## 4. Input / output contract

| | |
|---|---|
| Input | per-point `coord (N,3)` in metres, `color (N,3)`, `normal (N,3)` |
| Missing channels | set colour and normal to **zeros** (model card) |
| Intensity | not an input in the default model. Dropping it may cost you on retro-reflective classes (signs, plates) |
| Output | hierarchical point features. Upcast to the original points through the stored pooling inverse |
| Checkpoints | `utonia.pth` (inference package), `pretrain-utonia-v1m1-0-base_stagev2.pth` (Pointcept eval) |
| Sizes | base (54…576 ch), small, tiny (channel tables on the card) |

## 5. Pretraining data and domain match

| Factor | Automotive LiDAR | Utonia pretraining | Risk |
|---|---|---|---|
| Sensor pattern | spinning, ring structure, range-dependent density | includes outdoor LiDAR plus very different domains | **medium**: verify which outdoor datasets were used |
| Range | 0–200 m | mixed scales | medium: check grid size and coordinate normalisation |
| Channels | xyz + intensity (+ ring, time) | xyz + colour + normal | **high** for intensity-dependent classes |
| Motion | multi-sweep aggregation common | unknown | test single vs multi-sweep input |

**Verdict:** this is the most automotive-relevant Pointcept SSL encoder at the baseline date,
but automotive *detection* transfer is **not demonstrated by the card**. Treat it as a
transfer-learning candidate until Lab 06 measures it.

## 6. Official links and licence

- Weights + card: https://huggingface.co/Pointcept/Utonia (CC-BY-NC-4.0)
- Inference: https://github.com/Pointcept/Utonia · Pretraining: https://github.com/Pointcept/Pointcept
- Paper: https://arxiv.org/abs/2603.03283 · Project: https://pointcept.github.io/Utonia/

## 7. Run inference

Follow the official inference demo, since its install steps change. In outline: install the
Utonia package and spconv for your CUDA, load `utonia.pth`, build a dict with `coord`, `color`
and `normal` (zeros for the last two), grid-sample at the configured size, and run the
model. **Not executed in this repo** (no GPU, no network access to GitHub from the build container).

## 8. Adapting to automotive 3D detection: the gap, stated honestly

There is no off-the-shelf "Utonia + CenterPoint head" checkpoint. The adapter work is:

1. **Feature-to-BEV bridge:** scatter point features into a BEV grid (max- or mean-pool per cell), producing `(C, H, W)` like a pillar map.
2. **Detection head:** CenterPoint heatmap head on that BEV map (OpenPCDet / mmdet3d implementation).
3. **Training:** (a) frozen encoder with only bridge + head trained, then (b) partial unfreeze of the last stage at 0.1× LR.
4. **Control:** the same head on a PointPillars/VoxelNet backbone trained from scratch on the same labelled subset (1%, 10%, 100%).
5. **Metric:** nuScenes mAP/NDS, or KITTI AP_R40, **stratified by range**. A gain that exists only at 0–30 m is not a long-range win.

## 9. Export / quantisation / engine

None are offered. PTv3 relies on sparse convolution for embedding, serialisation and patch
attention. ONNX/TensorRT export is non-trivial and **not verified**. The realistic deployment
path is to **distil** into a smaller sparse-conv or pillar backbone (the card itself recommends
distilling small variants on in-domain data).

## 10. Benchmarks

No automotive detection numbers are published on the card. Nothing has been reproduced locally.

## 11. Failure modes and alternatives

- Features dominated by ground points: remove ground before probing, or report ground and objects separately.
- Scale mismatch: if grid sampling is too fine for 70 m range, you get OOM or very sparse patches.
- Alternatives: `concerto` (outdoor checkpoint, NC), `sonata` (licence unclear), `ptv3-pointcept` (supervised nuScenes checkpoint, MIT weights with nuScenes terms).

## 12. Sources

- https://huggingface.co/Pointcept/Utonia (model card read 2026-10-30)
