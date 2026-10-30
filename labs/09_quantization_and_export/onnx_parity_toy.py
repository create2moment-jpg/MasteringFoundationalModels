"""Lab 09 (CPU path): ONNX parity and INT8 PTQ on a tiny BEV heatmap head.

The full workflow (export -> parity -> quantise -> parity -> benchmark), at a
scale that runs on any CPU in seconds:

1. Build a 2-layer conv "centre-heatmap head" directly as an ONNX graph
   (stands in for ``torch.onnx.export`` output; same artifact type).
2. Check FP32 parity: ONNX Runtime vs an independent NumPy reference.
3. Static INT8 PTQ (QDQ, per-channel weights) with onnxruntime.quantization,
   calibrated on BEV maps from synthetic LiDAR scenes.
4. Compare INT8 vs FP32 on held-out scenes – and vs a *badly calibrated* INT8
   model – using numeric and task-level (peak agreement) metrics.
5. Latency of both models through the provenance-recording harness.

Requires: pip install -e ".[export]"   (onnx, onnxruntime; no torch)

Usage:
    python labs/09_quantization_and_export/onnx_parity_toy.py --out outputs/lab09
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from numpy.typing import NDArray
from onnx import TensorProto, helper, numpy_helper
from onnxruntime.quantization import (
    CalibrationDataReader,
    QuantFormat,
    QuantType,
    quantize_static,
)

from model_lab.benchmarking.harness import BenchmarkConfig, run_benchmark
from model_lab.data.synthetic import make_scene
from model_lab.data.voxelize import VoxelGrid, bev_occupancy

GRID = VoxelGrid((0.32, 0.32, 4.0), (0.0, -39.68, -3.0, 69.12, 39.68, 1.0))  # 216 x 248 BEV
C_IN, C_MID = 2, 8


def bev_input(seed: int) -> NDArray[np.float32]:
    density, height = bev_occupancy(make_scene(seed=seed).points, GRID)
    x = np.stack([np.log1p(density), (height + 3.0) / 4.0])  # roughly [0, ~3] and [0, 1]
    return x[None].astype(np.float32)  # (1, 2, H, W)


def build_model(path: Path, seed: int = 0) -> dict[str, NDArray[np.float32]]:
    rng = np.random.default_rng(seed)
    w1 = (rng.normal(size=(C_MID, C_IN, 3, 3)) * 0.4).astype(np.float32)
    b1 = (rng.normal(size=C_MID) * 0.1).astype(np.float32)
    w2 = (rng.normal(size=(1, C_MID, 1, 1)) * 0.5).astype(np.float32)
    b2 = np.array([-1.0], dtype=np.float32)
    nodes = [
        helper.make_node("Conv", ["bev", "w1", "b1"], ["c1"], pads=[1, 1, 1, 1]),
        helper.make_node("Relu", ["c1"], ["r1"]),
        helper.make_node("Conv", ["r1", "w2", "b2"], ["logits"]),
        helper.make_node("Sigmoid", ["logits"], ["heatmap"]),
    ]
    graph = helper.make_graph(
        nodes,
        "bev_heatmap_head",
        [helper.make_tensor_value_info("bev", TensorProto.FLOAT, [1, C_IN, "H", "W"])],
        [helper.make_tensor_value_info("heatmap", TensorProto.FLOAT, [1, 1, "H", "W"])],
        initializer=[
            numpy_helper.from_array(a, n)
            for n, a in (("w1", w1), ("b1", b1), ("w2", w2), ("b2", b2))
        ],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    onnx.save(model, path)
    return {"w1": w1, "b1": b1, "w2": w2, "b2": b2}


def numpy_reference(
    x: NDArray[np.float32], p: dict[str, NDArray[np.float32]]
) -> NDArray[np.float32]:
    """Independent float64 implementation of the same graph (the 'framework' side)."""
    xp = np.pad(x[0].astype(np.float64), ((0, 0), (1, 1), (1, 1)))
    _, hh, ww = x[0].shape
    c1 = np.zeros((C_MID, hh, ww))
    for dy in range(3):
        for dx in range(3):
            patch = xp[:, dy : dy + hh, dx : dx + ww]
            c1 += np.einsum("oc,chw->ohw", p["w1"][:, :, dy, dx], patch)
    r1 = np.maximum(c1 + p["b1"][:, None, None], 0)
    logits = np.einsum("oc,chw->ohw", p["w2"][:, :, 0, 0], r1) + p["b2"][:, None, None]
    return (1 / (1 + np.exp(-logits)))[None].astype(np.float32)


class Reader(CalibrationDataReader):
    def __init__(self, samples: list[NDArray[np.float32]]) -> None:
        self._it = iter([{"bev": s} for s in samples])

    def get_next(self) -> dict[str, NDArray[np.float32]] | None:
        return next(self._it, None)


def peaks(h: NDArray[np.float32], k: int = 20) -> set[int]:
    return set(np.argsort(h.ravel())[-k:].tolist())


def compare(name: str, ref: list[NDArray[np.float32]], out: list[NDArray[np.float32]]) -> None:
    a, b = np.concatenate([r.ravel() for r in ref]), np.concatenate([o.ravel() for o in out])
    cos = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))
    top = np.mean([len(peaks(r) & peaks(o)) / 20 for r, o in zip(ref, out, strict=True)])
    print(
        f"  {name:<34} max|diff| {np.abs(a - b).max():.2e}  mean|diff| {np.abs(a - b).mean():.2e}  "
        f"cosine {cos:.6f}  top-20 peak agreement {top:.2f}"
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=Path("outputs/lab09"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    fp32_path, int8_path, bad_path = (
        args.out / n for n in ("head_fp32.onnx", "head_int8.onnx", "head_int8_badcal.onnx")
    )

    params = build_model(fp32_path)
    calib = [bev_input(s) for s in range(100, 116)]  # calibration scenes
    test = [bev_input(s) for s in range(500, 508)]  # held-out scenes (never calibrate on test)

    sess = ort.InferenceSession(str(fp32_path), providers=["CPUExecutionProvider"])
    fp32 = [sess.run(None, {"bev": x})[0] for x in test]
    ref = [numpy_reference(x, params) for x in test]
    print("1) FP32 parity: ONNX Runtime vs NumPy reference (expect ~1e-6, float32 rounding)")
    compare("onnxruntime fp32 vs numpy", ref, fp32)

    common = dict(
        quant_format=QuantFormat.QDQ,
        per_channel=True,
        activation_type=QuantType.QUInt8,
        weight_type=QuantType.QInt8,
    )
    quantize_static(str(fp32_path), str(int8_path), Reader(calib), **common)  # type: ignore[arg-type]
    # Bad calibration: activations from empty scenes -> ranges far too small -> clipping.
    quantize_static(str(fp32_path), str(bad_path), Reader([c * 0.05 for c in calib]), **common)  # type: ignore[arg-type]

    int8_sess = ort.InferenceSession(str(int8_path), providers=["CPUExecutionProvider"])
    bad_sess = ort.InferenceSession(str(bad_path), providers=["CPUExecutionProvider"])
    int8 = [int8_sess.run(None, {"bev": x})[0] for x in test]
    bad = [bad_sess.run(None, {"bev": x})[0] for x in test]
    print("\n2) INT8 PTQ vs FP32 on held-out scenes")
    compare("int8 (16 representative scenes)", fp32, int8)
    compare("int8 (mis-scaled calibration)", fp32, bad)

    print("\n3) Artifacts (training ckpt != quantised weights != ONNX graph != engine):")
    for p in (fp32_path, int8_path):
        ops = sorted({n.op_type for n in onnx.load(p).graph.node})
        print(f"  {p.name:<26} {p.stat().st_size:>6} bytes  ops={ops}")

    print("\n4) Latency (CPUExecutionProvider; tiny model, so overhead-dominated)")
    for name, s in (("fp32", sess), ("int8", int8_sess)):
        rec = run_benchmark(
            BenchmarkConfig(
                model_id=f"toy-bev-head-{name}",
                input_desc="1x2x248x216",
                precision=name,
                runtime=f"onnxruntime {ort.__version__} CPU",
                warmup=5,
                iters=50,
            ),
            make_input=lambda i: test[i % len(test)],
            stages={"forward": lambda x, s=s: s.run(None, {"bev": x})[0]},
        )
        print(f"  {name}: p50 {rec.end_to_end.p50_ms:.3f} ms  p90 {rec.end_to_end.p90_ms:.3f} ms")


if __name__ == "__main__":
    main()
