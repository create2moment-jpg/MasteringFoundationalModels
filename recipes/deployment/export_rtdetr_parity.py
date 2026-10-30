"""Export RT-DETRv2 (HF) to ONNX and check numerical + task parity with PyTorch.

NOT EXECUTED in the repository build environment (no torch). Written against
public torch.onnx / transformers / onnxruntime APIs; verify on your versions.

Usage:
    python recipes/deployment/export_rtdetr_parity.py --images data/frames/*.jpg \
        --out outputs/rtdetr_onnx --height 640 --width 640 --opset 17

Then build a TensorRT engine from the ONNX (see README) and re-run the parity
section against the engine outputs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from PIL import Image
from transformers import AutoImageProcessor, RTDetrV2ForObjectDetection

MODEL_ID = "PekingU/rtdetr_v2_r50vd"


class Wrapper(torch.nn.Module):
    """Expose plain tensors (logits, boxes) – ONNX cannot return HF ModelOutput objects."""

    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, pixel_values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        out = self.model(pixel_values=pixel_values)
        return out.logits, out.pred_boxes


def topk_dets(
    logits: np.ndarray, boxes: np.ndarray, k: int = 100
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """RT-DETR decoding: sigmoid, top-k over (query, class) – mirrors HF post-processing."""
    prob = 1 / (1 + np.exp(-logits[0]))
    flat = prob.ravel()
    idx = np.argsort(-flat)[:k]
    q, c = np.divmod(idx, prob.shape[1])
    return flat[idx], c, boxes[0][q]


def box_iou_cxcywh(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    def xyxy(x: np.ndarray) -> np.ndarray:
        return np.concatenate([x[:, :2] - x[:, 2:] / 2, x[:, :2] + x[:, 2:] / 2], 1)

    a, b = xyxy(a), xyxy(b)
    lt, rb = np.maximum(a[:, None, :2], b[None, :, :2]), np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.prod(np.clip(rb - lt, 0, None), axis=2)
    area = lambda x: np.prod(x[:, 2:] - x[:, :2], axis=1)  # noqa: E731
    return inter / (area(a)[:, None] + area(b)[None] - inter + 1e-9)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--images", nargs="+", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--revision", default="main")
    ap.add_argument("--height", type=int, default=640)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--score", type=float, default=0.3)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    proc = AutoImageProcessor.from_pretrained(MODEL_ID, revision=args.revision)
    model = RTDetrV2ForObjectDetection.from_pretrained(MODEL_ID, revision=args.revision).eval()
    wrapped = Wrapper(model)
    onnx_path = args.out / "rtdetrv2_r50vd.onnx"
    dummy = torch.zeros(1, 3, args.height, args.width)
    torch.onnx.export(
        wrapped,
        (dummy,),
        str(onnx_path),
        opset_version=args.opset,
        input_names=["pixel_values"],
        output_names=["logits", "pred_boxes"],
        dynamic_axes={
            "pixel_values": {0: "batch"},
            "logits": {0: "batch"},
            "pred_boxes": {0: "batch"},
        },
    )
    sha = hashlib.sha256(onnx_path.read_bytes()).hexdigest()
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])

    report = []
    for path in args.images:
        pv = proc(
            images=Image.open(path).convert("RGB"),
            size={"height": args.height, "width": args.width},
            return_tensors="pt",
        )["pixel_values"]
        with torch.inference_mode():
            t_logits, t_boxes = (t.numpy() for t in wrapped(pv))
        o_logits, o_boxes = sess.run(None, {"pixel_values": pv.numpy()})
        ts, tc, tb = topk_dets(t_logits, t_boxes)
        os_, oc, ob = topk_dets(o_logits, o_boxes)
        keep_t, keep_o = ts > args.score, os_ > args.score
        iou = (
            box_iou_cxcywh(tb[keep_t], ob[keep_o])
            if keep_t.any() and keep_o.any()
            else np.zeros((0, 0))
        )
        same_cls = (tc[keep_t][:, None] == oc[keep_o][None]) if iou.size else iou.astype(bool)
        matched = (
            float(((iou > 0.95) & same_cls).any(1).mean())
            if iou.size
            else float(keep_t.sum() == keep_o.sum())
        )
        report.append(
            {
                "image": path.name,
                "max_abs_logit_diff": float(np.abs(t_logits - o_logits).max()),
                "max_abs_box_diff": float(np.abs(t_boxes - o_boxes).max()),
                "n_dets_torch": int(keep_t.sum()),
                "n_dets_onnx": int(keep_o.sum()),
                "frac_torch_dets_matched_iou95_same_class": matched,
            }
        )
        print(report[-1])
    meta = {
        "model": MODEL_ID,
        "revision": args.revision,
        "opset": args.opset,
        "input_hw": [args.height, args.width],
        "onnx_sha256": sha,
        "torch": torch.__version__,
        "onnxruntime": ort.__version__,
        "results": report,
    }
    (args.out / "parity_report.json").write_text(json.dumps(meta, indent=2))
    print(f"wrote {onnx_path} (sha256 {sha[:12]}…) and parity_report.json")


if __name__ == "__main__":
    main()
