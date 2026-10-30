# Recipe: export → parity → TensorRT engine → parity → benchmark

Worked example: RT-DETRv2-R50 (2D). The LiDAR and fusion variants follow at the end.
**Nothing in Part B was executed in this repo.** The executed, CPU-scale version of the same
loop is [`labs/09_quantization_and_export`](../../labs/09_quantization_and_export/README.md).

## Artifacts: never conflate these

| # | Artifact | Example | Portable? |
|---|---|---|---|
| 1 | Training checkpoint | `model.safetensors` (HF), `.pth` | across machines with the same code |
| 2 | Quantised weights / QDQ model | `rtdetr_int8_qdq.onnx` | runtime-dependent |
| 3 | Portable graph | `rtdetrv2_r50vd.onnx` (opset 17) | yes, across ONNX runtimes |
| 4 | Compiled engine | `rtdetr_fp16.plan` | **no**: tied to GPU arch, TRT version and profile |
| 5 | Runtime | TensorRT 10.x + plugins, ORT + EP | installed per machine |

A prebuilt `.plan` downloaded from the internet almost never matches your GPU and TRT
version. The reliable deliverable is a **reproducible build command**.

## Steps

```bash
# 1. export + parity vs PyTorch (CPU is fine)
python recipes/deployment/export_rtdetr_parity.py --images data/frames/*.jpg --out outputs/rtdetr_onnx

# 2. build FP16 engine (on the target GPU)
trtexec --onnx=outputs/rtdetr_onnx/rtdetrv2_r50vd.onnx --saveEngine=outputs/rtdetr_fp16.plan \
        --fp16 --minShapes=pixel_values:1x3x640x640 --optShapes=pixel_values:1x3x640x640 \
        --maxShapes=pixel_values:4x3x640x640 --verbose 2>&1 | tee outputs/trtexec_fp16.log

# 3. INT8: either QDQ-ONNX from modelopt / ORT quantisation (explicit) or TensorRT
#    implicit calibration with a calibrator over ~500 representative DRIVING frames.
#    Keep the decoder in FP16 if per-class AP drops > 1 point.

# 4. parity against the engine: reuse the top-k decoding + IoU-0.95 matching from step 1;
#    then AP on a fixed 500-frame val subset for torch vs fp16 vs int8.
```

Record for every engine: GPU model, driver, CUDA, TensorRT version, precision flags, shape
profile, calibration set ID, full command, ONNX SHA-256, engine SHA-256, and the parity report.

## Acceptance criteria (suggested)

| check | FP16 | INT8 |
|---|---|---|
| max abs logit diff | < 1e-2 | report only |
| torch detections matched at IoU > 0.95, same class | > 99% | > 95% |
| AP delta on fixed val subset | < 0.3 | < 1.0 (per class too) |

## Known export traps (2D)

- HF `ModelOutput` cannot be exported, so wrap it to return tensors (done in the script).
- `grid_sample` / deformable attention: requires opset ≥ 16. Check TensorRT layer support for your version.
- Pre-processing (resize, normalise) lives outside the graph and must match bit-for-bit (PIL vs OpenCV resize differ).

## LiDAR / fusion specifics

- Voxelisation, pillar scatter, sweep accumulation and NMS sit **outside** the graph. Ship them as CUDA kernels or C++ and parity-test them separately (`model_lab.data.voxelize` is the slow reference).
- Sparse convolution: not an ONNX standard op. Use vendor plugins (NVIDIA Lidar_AI_Solution `libspconv`, Autoware `autoware_tensorrt_plugins`) or switch to a pillar backbone.
- BEVFusion: the Autoware ONNX models plus a first-launch engine build are the most complete public path (see [`models/cards/bevfusion-mit.md`](../../models/cards/bevfusion-mit.md)).
