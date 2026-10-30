# 05: Quantisation, acceleration and deployment

## Artifact chain: five things that are not the same

`training checkpoint` → `converted / quantised weights` → `portable graph (ONNX)` → `compiled engine (TensorRT plan, OpenVINO IR blob)` → `runtime library + plugins`.
The registry stores a status for each (`artifacts.*`). Most public models offer only the
first. Engines are hardware- and version-specific, so the deliverable is a build recipe.

## Engine / runtime matrix

| runtime | best for | precision | custom-op story | notes |
|---|---|---|---|---|
| PyTorch eager | research, debugging | FP32/BF16/FP16 | n/a | baseline for parity |
| `torch.compile` | training & server inference speedups | same | graph breaks on dynamic Python | check recompilation on shape changes |
| ONNX + ONNX Runtime | portable inference, CPU/GPU EPs | FP32/FP16/INT8 (QDQ) | custom ops via C++ | executed CPU example: Lab 09 A |
| TensorRT | NVIDIA GPUs incl. Jetson/DRIVE | FP32/FP16/BF16/INT8/FP8 (arch-dependent) | plugins (C++/CUDA) | engines tied to GPU arch + TRT version + profile |
| Torch-TensorRT | stay in PyTorch, compile subgraphs | as TRT | falls back to Torch for unsupported ops | convenient for partial support |
| OpenVINO | Intel CPU/GPU/NPU | FP32/FP16/INT8 | extensions | check supported-model list per release |
| MMDeploy | OpenMMLab models (mmdet, mmdet3d) | as backend | ships some custom ops | version coupling with mm* packages |
| Triton / DeepStream / ROS 2 | serving, video pipelines, robotics integration | n/a | n/a | Autoware runs TensorRT inside ROS 2 nodes |

FP8 and sub-INT8 formats apply only where the hardware, TensorRT version and layer types
support them. Verify against the TensorRT "quantized types" documentation for your version.

## Quantisation concepts (short)

- **Dynamic vs static:** dynamic quantises activations at runtime (good for MatMul-heavy CPU models). Static uses calibrated ranges (the norm for conv detectors on GPU and NPU).
- **Weight-only vs weight+activation:** weight-only saves memory but rarely speeds up conv detectors. W8A8 gives real INT8 throughput.
- **PTQ vs QAT:** PTQ needs about 100–1000 representative frames. QAT fine-tunes with fake-quant and is used when PTQ costs more than your accuracy budget.
- **Per-channel vs per-tensor:** per-channel weights are almost always better for convs.
- **Other levers:** sparsity (structured 2:4 needs HW support), pruning, distillation, kernel fusion, input-resolution changes, batching. Each trades latency against memory and accuracy differently. Always measure p99 end-to-end.

## Known export traps

| trap | where | mitigation |
|---|---|---|
| NMS in post-processing | YOLO, anchor heads, LiDAR heads | export without NMS + runtime NMS, or TRT EfficientNMS plugin; parity-check decoded boxes |
| dynamic shapes | variable image size, point counts | fixed profiles; pad points to max voxels |
| sparse conv | SECOND, CenterPoint, BEVFusion LiDAR branch | vendor libs (`libspconv`), Autoware plugins, or pillar backbones |
| voxelisation / scatter outside graph | all LiDAR | reference implementation + parity test (`model_lab.data.voxelize`) |
| deformable attention / grid_sample | DETR family, BEVFormer | opset ≥ 16; TRT plugin or native support check |
| BEV pooling | LSS / BEVFusion | custom CUDA kernel; recompute geometry on calibration change |
| numerical drift | FP16 softmax/sigmoid on large logits, INT8 decoders | keep sensitive layers FP16; task-level parity |
| pre-processing mismatch | resize/normalise outside graph | same library and interpolation; bit-exact test |

## Parity protocol (minimum)

1. Same inputs (saved tensors) through framework and exported model.
2. Tensor-level: max/mean abs diff and cosine per output.
3. **Task-level:** decoded detections matched at IoU > 0.95 with the same class, plus AP on a fixed subset.
4. Record versions, hardware, command and hashes in a `BenchmarkRecord`.

Lab 09 A shows why step 3 is not optional: cosine 0.995 with 8% heatmap peak agreement.

## Model-specific paths

- 2D: [`recipes/deployment`](../recipes/deployment/README.md) (RT-DETRv2 → ONNX → TensorRT).
- Fusion: [`models/cards/bevfusion-mit.md`](../models/cards/bevfusion-mit.md) §9 (Autoware ONNX + first-launch engine build; NVIDIA CUDA-BEVFusion).
