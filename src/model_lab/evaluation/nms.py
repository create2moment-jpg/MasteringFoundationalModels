"""Greedy non-maximum suppression (reference implementations).

NMS is the most common *export trap* for detectors: it is data-dependent,
often lives outside the ONNX graph, and differs between frameworks
(class-aware vs agnostic, IoU vs distance, pre/post top-k). Query-based
detectors (DETR family, RT-DETR) are NMS-free by design, which is one reason
they export more cleanly.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray

from model_lab.evaluation.iou import iou_2d, iou_bev


def _greedy(
    scores: NDArray[np.floating],
    overlap: Callable[[int, NDArray[np.int64]], NDArray[np.float64]],
    thresh: float,
) -> NDArray[np.int64]:
    order = np.argsort(-np.asarray(scores), kind="stable")
    keep: list[int] = []
    while len(order):
        i = int(order[0])
        keep.append(i)
        rest = order[1:]
        if len(rest) == 0:
            break
        order = rest[overlap(i, rest) <= thresh]
    return np.array(keep, dtype=np.int64)


def nms_2d(
    boxes: NDArray[np.floating], scores: NDArray[np.floating], iou_thresh: float = 0.5
) -> NDArray[np.int64]:
    """Indices kept by class-agnostic greedy NMS on ``[x1, y1, x2, y2]`` boxes, by descending score."""
    b = np.asarray(boxes, dtype=np.float64)
    return _greedy(scores, lambda i, rest: iou_2d(b[i : i + 1], b[rest])[0], iou_thresh)


def nms_bev(
    boxes: NDArray[np.floating], scores: NDArray[np.floating], iou_thresh: float = 0.1
) -> NDArray[np.int64]:
    """Rotated-BEV greedy NMS on 7-DoF boxes (typical LiDAR thresholds are 0.01–0.2)."""
    b = np.asarray(boxes, dtype=np.float64)
    return _greedy(scores, lambda i, rest: iou_bev(b[i : i + 1], b[rest])[0], iou_thresh)


def circle_nms(
    boxes: NDArray[np.floating], scores: NDArray[np.floating], radius: float
) -> NDArray[np.int64]:
    """CenterPoint-style NMS by BEV centre distance (cheap; used for small classes)."""
    b = np.asarray(boxes, dtype=np.float64)

    def close(i: int, rest: NDArray[np.int64]) -> NDArray[np.float64]:
        d = np.hypot(b[rest, 0] - b[i, 0], b[rest, 1] - b[i, 1])
        return np.asarray(d < radius, dtype=np.float64)

    return _greedy(scores, close, 0.5)
