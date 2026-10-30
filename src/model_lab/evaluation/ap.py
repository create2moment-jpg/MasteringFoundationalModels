"""Detection matching and average precision (AP) – transparent reference code.

Why a reference implementation? Because "mAP" is not one number. It depends
on the matching criterion (2D IoU, BEV IoU, 3D IoU, or centre distance), the
threshold, the interpolation rule, difficulty / range filters and ignore
regions. Reading ~150 lines that implement the variants side by side is the
fastest way to understand why KITTI, COCO and nuScenes numbers are not
comparable.

For *reportable* numbers always use the official tooling (pycocotools, the
KITTI devkit, nuscenes-devkit, waymo-open-dataset metrics). These functions
are for learning, unit tests and quick internal comparisons on identical
protocols.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import Enum

import numpy as np
from numpy.typing import NDArray

from model_lab.evaluation.iou import iou_2d, iou_3d, iou_bev

FloatArray = NDArray[np.float64]


class Interp(str, Enum):
    COCO_101 = "coco_101"  # mean precision at recall 0, 0.01, ..., 1 (monotone envelope)
    KITTI_40 = "kitti_40"  # R40: recall 1/40, ..., 1 (KITTI since 2019)
    VOC_ALL = "voc_all_points"  # exact area under the monotone PR envelope
    NUSCENES = "nuscenes"  # 101-pt, drop recall<0.1, subtract min precision 0.1, renormalise


class Criterion(str, Enum):
    IOU_2D = "iou_2d"
    IOU_BEV = "iou_bev"
    IOU_3D = "iou_3d"
    CENTER_DIST = "center_distance"  # nuScenes: BEV centre distance in metres (lower is better)


@dataclass(frozen=True)
class FrameDetections:
    """Detections and ground truth for one frame and one class."""

    pred_boxes: NDArray[np.floating]
    pred_scores: NDArray[np.floating]
    gt_boxes: NDArray[np.floating]


@dataclass(frozen=True)
class APResult:
    ap: float
    precision: FloatArray
    recall: FloatArray
    num_gt: int
    num_pred: int


def _similarity(
    criterion: Criterion,
) -> Callable[[NDArray[np.floating], NDArray[np.floating]], FloatArray]:
    if criterion is Criterion.IOU_2D:
        return iou_2d
    if criterion is Criterion.IOU_BEV:
        return iou_bev
    if criterion is Criterion.IOU_3D:
        return iou_3d

    def neg_center_dist(a: NDArray[np.floating], b: NDArray[np.floating]) -> FloatArray:
        a2, b2 = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
        d = np.hypot(a2[:, None, 0] - b2[None, :, 0], a2[:, None, 1] - b2[None, :, 1])
        return np.asarray(-d, dtype=np.float64)  # higher = better, so threshold is -max_distance

    return neg_center_dist


def match_frame(
    frame: FrameDetections, criterion: Criterion, threshold: float
) -> tuple[FloatArray, NDArray[np.bool_]]:
    """Greedy matching by descending score.

    Each prediction is matched to the *best still-unmatched* GT whose similarity
    passes the threshold (COCO / nuScenes behaviour).

    Returns:
        ``(scores, is_true_positive)`` for every prediction in the frame.
    """
    scores = np.asarray(frame.pred_scores, dtype=np.float64)
    tp = np.zeros(len(scores), dtype=bool)
    n_gt = len(frame.gt_boxes)
    if len(scores) == 0 or n_gt == 0:
        return scores, tp
    sim = _similarity(criterion)(frame.pred_boxes, frame.gt_boxes)
    thr = -threshold if criterion is Criterion.CENTER_DIST else threshold
    taken = np.zeros(n_gt, dtype=bool)
    for i in np.argsort(-scores, kind="stable"):
        cand = np.where(~taken & (sim[i] >= thr), sim[i], -np.inf)
        j = int(np.argmax(cand))
        if np.isfinite(cand[j]):
            taken[j] = True
            tp[i] = True
    return scores, tp


def precision_recall(
    frames: Sequence[FrameDetections], criterion: Criterion, threshold: float
) -> tuple[FloatArray, FloatArray, int]:
    """Accumulate matches over frames into a PR curve sorted by descending score."""
    all_scores, all_tp = [], []
    num_gt = 0
    for f in frames:
        s, tp = match_frame(f, criterion, threshold)
        all_scores.append(s)
        all_tp.append(tp)
        num_gt += len(f.gt_boxes)
    if not all_scores:
        return np.zeros(0), np.zeros(0), num_gt
    scores = np.concatenate(all_scores)
    tps = np.concatenate(all_tp)
    order = np.argsort(-scores, kind="stable")
    tps = tps[order]
    ctp = np.cumsum(tps)
    cfp = np.cumsum(~tps)
    recall = ctp / max(num_gt, 1)
    precision = ctp / np.maximum(ctp + cfp, 1)
    return precision, recall, num_gt


def average_precision(
    precision: FloatArray, recall: FloatArray, interp: Interp = Interp.COCO_101
) -> float:
    """Area under a PR curve using the given interpolation rule."""
    if len(precision) == 0:
        return 0.0
    # monotone (non-increasing) precision envelope
    env = np.maximum.accumulate(precision[::-1])[::-1]

    def sample(points: FloatArray) -> FloatArray:
        idx = np.searchsorted(recall, points, side="left")
        return np.where(idx < len(env), env[np.minimum(idx, len(env) - 1)], 0.0)

    if interp is Interp.COCO_101:
        return float(sample(np.linspace(0, 1, 101)).mean())
    if interp is Interp.KITTI_40:
        return float(sample(np.linspace(1 / 40, 1, 40)).mean())
    if interp is Interp.NUSCENES:
        min_recall, min_precision = 0.1, 0.1
        prec = sample(np.linspace(0, 1, 101))[round(100 * min_recall) + 1 :]
        prec = np.clip(prec - min_precision, 0, None)
        return float(prec.mean() / (1 - min_precision))
    # VOC all-points
    r = np.concatenate([[0.0], recall])
    p = np.concatenate([[env[0]], env])
    return float(np.sum((r[1:] - r[:-1]) * p[1:]))


def evaluate_ap(
    frames: Sequence[FrameDetections],
    criterion: Criterion,
    threshold: float,
    interp: Interp = Interp.COCO_101,
) -> APResult:
    p, r, n_gt = precision_recall(frames, criterion, threshold)
    return APResult(
        ap=average_precision(p, r, interp),
        precision=p,
        recall=r,
        num_gt=n_gt,
        num_pred=int(sum(len(f.pred_scores) for f in frames)),
    )


def range_filter(frame: FrameDetections, r_min: float, r_max: float) -> FrameDetections:
    """Keep predictions and GT whose BEV centre range is in ``[r_min, r_max)``.

    Distance-stratified AP is essential for driving: a model can look strong
    overall while failing beyond 50 m, where LiDAR returns are sparse and
    camera objects are a few pixels tall. Note: filtering predictions and GT
    independently at bin edges slightly penalises boxes straddling a boundary;
    official toolkits handle this with ignore regions.
    """

    def in_range(b: NDArray[np.floating]) -> NDArray[np.bool_]:
        b = np.asarray(b, dtype=np.float64).reshape(-1, 7)
        rng = np.hypot(b[:, 0], b[:, 1])
        return np.asarray((rng >= r_min) & (rng < r_max), dtype=bool)

    pm, gm = in_range(frame.pred_boxes), in_range(frame.gt_boxes)
    return FrameDetections(
        np.asarray(frame.pred_boxes)[pm],
        np.asarray(frame.pred_scores)[pm],
        np.asarray(frame.gt_boxes)[gm],
    )
