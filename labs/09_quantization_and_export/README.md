# Lab 09: export, numerical parity and INT8 PTQ

**Objective:** practise the deployment loop (export → parity → quantise → parity → benchmark) and
learn to keep the artifact types apart: training checkpoint ≠ quantised weights ≠ ONNX graph
≠ compiled engine ≠ runtime.

## Part A: toy BEV head on ONNX Runtime CPU (executed 2026-10-30)

```bash
pip install -e ".[export]"        # onnx + onnxruntime only, no torch needed
python labs/09_quantization_and_export/onnx_parity_toy.py --out outputs/lab09
```

As executed (onnx 1.23.2, onnxruntime 1.31.1, CPUExecutionProvider):

| comparison (8 held-out synthetic scenes) | max abs diff | cosine | top-20 peak agreement |
|---|---|---|---|
| ORT FP32 vs NumPy float64 reference | 1.0e-7 | 1.000000 | 1.00 |
| INT8 (calibrated on 16 representative scenes) vs FP32 | 1.3e-2 | 1.000000 | 0.91 |
| INT8 (mis-scaled calibration) vs FP32 | 2.3e-1 | 0.994567 | **0.08** |

**The key lesson:** a 0.995 cosine similarity *looks* fine, yet the badly calibrated model
puts its heatmap peaks (and therefore detections) in the wrong places. Parity checks must
include a **task-level** comparison (decoded boxes, matched IoU, AP on a fixed subset), not only
tensor similarity.

Latency on this toy (p50 about 1.0 ms FP32 vs 0.85 ms INT8) is dominated by overhead. Do not
generalise speedups from tiny models.

## Part B: real model (GPU + TensorRT, **not executed here**)

Recipe: [`recipes/deployment/README.md`](../../recipes/deployment/README.md), with RT-DETRv2 →
ONNX → TensorRT FP16/INT8 and a parity script outline. Record: GPU, driver, CUDA, TensorRT
version, precision, optimisation profile, calibration set, command line, and engine SHA-256.

## Quiz

1. Why does the INT8 model still contain `Conv` (not `QLinearConv`)? What will TensorRT do with QDQ nodes?
2. Why must calibration data come from the deployment domain and never from the test split?
3. Name three things that live outside the ONNX graph in a LiDAR detector deployment.

**Stretch:** set `per_channel=False` and compare peak agreement. Then exclude the last conv from
quantisation (`nodes_to_exclude`) and compare again.
