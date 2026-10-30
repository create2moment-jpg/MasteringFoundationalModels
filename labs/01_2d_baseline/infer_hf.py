"""Lab 01: RT-DETRv2 inference on driving images with a COCO->driving class map.

NOT EXECUTED in the repository build environment (no torch / no network). It is
written against the public transformers RT-DETRv2 API; run it and fix anything
your installed versions disagree with, then record the versions in your notes.

Requires: torch (install for your CUDA first), then  pip install -e ".[vision]"

Usage:
    python labs/01_2d_baseline/infer_hf.py --images path/to/frames/*.jpg --out outputs/lab01
    python labs/01_2d_baseline/infer_hf.py --images img.jpg --revision <commit-sha> --device cuda
Outputs one JSON per image (boxes in pixels, xyxy) plus timing.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import yaml
from PIL import Image
from transformers import AutoImageProcessor, RTDetrV2ForObjectDetection

MODEL_ID = "PekingU/rtdetr_v2_r50vd"
CLASS_MAP = Path(__file__).resolve().parents[2] / "configs" / "class_maps" / "coco_to_driving.yaml"


def load_class_map(model_id2label: dict[int, str]) -> dict[int, str]:
    cfg = yaml.safe_load(CLASS_MAP.read_text())
    out: dict[int, str] = {}
    for coco_id, (coco_name, driving) in cfg["mapping"].items():
        actual = model_id2label.get(int(coco_id))
        if actual != coco_name:  # guard against off-by-one / 91-vs-80 id schemes
            raise ValueError(
                f"class map mismatch at id {coco_id}: map says {coco_name!r}, model says {actual!r}"
            )
        out[int(coco_id)] = driving
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--images", nargs="+", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("outputs/lab01"))
    ap.add_argument("--revision", default="main", help="pin a commit sha for reproducibility")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--threshold", type=float, default=0.3)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    processor = AutoImageProcessor.from_pretrained(MODEL_ID, revision=args.revision)
    model = (
        RTDetrV2ForObjectDetection.from_pretrained(MODEL_ID, revision=args.revision)
        .to(args.device)
        .eval()
    )
    cmap = load_class_map(model.config.id2label)
    print(f"model {MODEL_ID}@{args.revision} on {args.device}; processor size={processor.size}")

    for path in args.images:
        image = Image.open(path).convert("RGB")
        t0 = time.perf_counter()
        inputs = processor(images=image, return_tensors="pt").to(args.device)
        t1 = time.perf_counter()
        with torch.inference_mode():
            outputs = model(**inputs)
        if args.device.startswith("cuda"):
            torch.cuda.synchronize()
        t2 = time.perf_counter()
        res = processor.post_process_object_detection(
            outputs, target_sizes=torch.tensor([image.size[::-1]]), threshold=args.threshold
        )[0]
        t3 = time.perf_counter()
        dets = [
            {
                "label": cmap[int(lab)],
                "coco_label": model.config.id2label[int(lab)],
                "score": round(float(s), 4),
                "box_xyxy": [round(float(v), 1) for v in b],
            }
            for s, lab, b in zip(res["scores"], res["labels"], res["boxes"], strict=True)
            if int(lab) in cmap
        ]
        record = {
            "image": path.name,
            "image_size_wh": image.size,
            "model": MODEL_ID,
            "revision": args.revision,
            "timing_ms": {
                "preprocess": 1e3 * (t1 - t0),
                "forward": 1e3 * (t2 - t1),
                "postprocess": 1e3 * (t3 - t2),
            },
            "detections": dets,
        }
        (args.out / f"{path.stem}.json").write_text(json.dumps(record, indent=2))
        fwd = record["timing_ms"]["forward"]
        print(f"{path.name}: {len(dets)} driving-class detections, forward {fwd:.1f} ms")


if __name__ == "__main__":
    main()
