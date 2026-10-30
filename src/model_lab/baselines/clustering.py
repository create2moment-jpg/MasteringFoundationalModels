"""Non-learned LiDAR object baseline: ground removal + BEV clustering + box fitting.

Why have a classical baseline in a foundation-model repo? Because it:

1. exercises the *whole* evaluation pipeline (frames, boxes, matching, AP,
   range stratification, latency) with zero dependencies;
2. gives a floor any learned detector must clear on your data;
3. shows *why* learned detectors exist: LiDAR sees only the sensor-facing
   surface (an "L" shape), so geometric box fitting under-estimates size and
   mis-places the centre, collapsing AP at strict IoU (0.7) while looking fine
   under centre-distance matching.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from model_lab.data.geometry import limit_period


@dataclass(frozen=True)
class ClusterConfig:
    ground_z: float = -1.73  # sensor height below LiDAR origin
    ground_margin: float = 0.25  # points within this height of the ground are removed
    max_range: float = 70.0
    cell: float = 0.3  # BEV grid cell for connectivity (m)
    min_points: int = 5


@dataclass(frozen=True)
class Detections:
    boxes: NDArray[np.float64]  # (K, 7) lab convention
    scores: NDArray[np.float64]  # (K,)
    num_points: NDArray[np.int64]  # (K,)


def remove_ground(points: NDArray[np.floating], cfg: ClusterConfig) -> NDArray[np.float32]:
    """Flat-ground removal. Real roads need RANSAC / grid-wise fitting (slopes, kerbs)."""
    p = np.asarray(points, dtype=np.float32)
    r = np.hypot(p[:, 0], p[:, 1])
    keep = (p[:, 2] > cfg.ground_z + cfg.ground_margin) & (r < cfg.max_range)
    return p[keep]


def bev_connected_components(xy: NDArray[np.floating], cell: float) -> NDArray[np.int64]:
    """Label points by 8-connected components of occupied BEV cells (-1 never occurs)."""
    if len(xy) == 0:
        return np.zeros(0, dtype=np.int64)
    ij = np.floor(np.asarray(xy) / cell).astype(np.int64)
    cells, inverse = np.unique(ij, axis=0, return_inverse=True)
    inverse = inverse.reshape(-1)
    index = {(int(a), int(b)): k for k, (a, b) in enumerate(cells)}
    comp = np.full(len(cells), -1, dtype=np.int64)
    n = 0
    for start in range(len(cells)):
        if comp[start] >= 0:
            continue
        comp[start] = n
        queue = deque([start])
        while queue:
            a, b = cells[queue.popleft()]
            for da in (-1, 0, 1):
                for db in (-1, 0, 1):
                    k = index.get((int(a + da), int(b + db)))
                    if k is not None and comp[k] < 0:
                        comp[k] = n
                        queue.append(k)
        n += 1
    return comp[inverse]


def fit_box(points: NDArray[np.floating], ground_z: float) -> NDArray[np.float64]:
    """PCA-oriented bounding box of a cluster, extended down to the ground plane.

    Yaw is ambiguous by pi (no heading cue) and, for L-shaped returns, PCA
    picks a diagonal-ish axis – both are documented failure modes.
    """
    p = np.asarray(points, dtype=np.float64)
    xy = p[:, :2]
    mu = xy.mean(axis=0)
    if len(p) >= 3:
        cov = np.cov((xy - mu).T)
        evals, evecs = np.linalg.eigh(cov)
        major = evecs[:, int(np.argmax(evals))]
        yaw = float(np.arctan2(major[1], major[0]))
    else:
        yaw = 0.0
    c, s = np.cos(-yaw), np.sin(-yaw)
    local = (xy - mu) @ np.array([[c, s], [-s, c]])
    lo, hi = local.min(axis=0), local.max(axis=0)
    mid_local = (lo + hi) / 2
    cc, ss = np.cos(yaw), np.sin(yaw)
    center_xy = mu + np.array(
        [cc * mid_local[0] - ss * mid_local[1], ss * mid_local[0] + cc * mid_local[1]]
    )
    l, w = np.maximum(hi - lo, 0.1)
    z_top = float(p[:, 2].max())
    h = max(z_top - ground_z, 0.1)
    return np.array(
        [center_xy[0], center_xy[1], ground_z + h / 2, l, w, h, float(limit_period(np.array(yaw)))]
    )


def detect(points: NDArray[np.floating], cfg: ClusterConfig | None = None) -> Detections:
    """Run the full baseline on one ``(N, 3+)`` LiDAR sweep."""
    cfg = cfg or ClusterConfig()
    obj = remove_ground(points, cfg)
    labels = bev_connected_components(obj[:, :2], cfg.cell)
    boxes, scores, counts = [], [], []
    n_clusters = int(np.max(labels)) + 1 if len(labels) else 0
    for k in range(n_clusters):
        member = obj[labels == k]
        if len(member) < cfg.min_points:
            continue
        boxes.append(fit_box(member, cfg.ground_z))
        counts.append(len(member))
        scores.append(1.0 - np.exp(-len(member) / 50.0))  # monotone in support, in (0, 1)
    return Detections(
        boxes=np.array(boxes, dtype=np.float64).reshape(-1, 7),
        scores=np.array(scores, dtype=np.float64),
        num_points=np.array(counts, dtype=np.int64),
    )
