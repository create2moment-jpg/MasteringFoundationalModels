"""IoU for 2D axis-aligned, BEV rotated and 3D rotated boxes (NumPy reference).

Rotated IoU is computed exactly by convex-polygon clipping
(Sutherland–Hodgman), which is what CUDA kernels in OpenPCDet/mmdet3d
approximate. 3D IoU here is "BEV intersection x vertical overlap", the
definition used by the KITTI 3D AP metric for upright (yaw-only) boxes.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from model_lab.data.geometry import box_bev_corners

FloatArray = NDArray[np.float64]


def iou_2d(a: NDArray[np.floating], b: NDArray[np.floating]) -> FloatArray:
    """Pairwise IoU of ``(N, 4)`` and ``(M, 4)`` ``[x1, y1, x2, y2]`` boxes -> ``(N, M)``."""
    a = np.asarray(a, dtype=np.float64).reshape(-1, 4)
    b = np.asarray(b, dtype=np.float64).reshape(-1, 4)
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = np.clip(rb - lt, 0, None)
    inter = wh[..., 0] * wh[..., 1]
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter
    return np.where(union > 0, inter / np.where(union > 0, union, 1), 0.0)


def _polygon_area(poly: FloatArray) -> float:
    if len(poly) < 3:
        return 0.0
    x, y = poly[:, 0], poly[:, 1]
    return float(0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def _clip(subject: FloatArray, clipper: FloatArray) -> FloatArray:
    """Sutherland–Hodgman: intersection of a convex polygon with a CCW convex clipper."""
    out = subject
    for i in range(len(clipper)):
        if len(out) == 0:
            break
        a, b = clipper[i], clipper[(i + 1) % len(clipper)]
        edge = b - a

        def side(p: FloatArray, a: FloatArray = a, edge: FloatArray = edge) -> float:
            return float(edge[0] * (p[1] - a[1]) - edge[1] * (p[0] - a[0]))

        inp, out_pts = out, []
        for j in range(len(inp)):
            cur, prev = inp[j], inp[j - 1]
            sc, sp = side(cur), side(prev)
            if sc >= 0:
                if sp < 0:
                    out_pts.append(prev + (cur - prev) * (sp / (sp - sc)))
                out_pts.append(cur)
            elif sp >= 0:
                out_pts.append(prev + (cur - prev) * (sp / (sp - sc)))
        out = np.array(out_pts).reshape(-1, 2)
    return out


def bev_intersection(a: NDArray[np.floating], b: NDArray[np.floating]) -> FloatArray:
    """Pairwise BEV footprint intersection area of 7-DoF boxes -> ``(N, M)`` m²."""
    ca, cb = box_bev_corners(a), box_bev_corners(b)
    inter = np.zeros((len(ca), len(cb)))
    for i, pa in enumerate(ca):
        for j, pb in enumerate(cb):
            inter[i, j] = _polygon_area(_clip(pa, pb))
    return inter


def iou_bev(a: NDArray[np.floating], b: NDArray[np.floating]) -> FloatArray:
    """Pairwise rotated BEV IoU of ``(N, 7)`` and ``(M, 7)`` boxes."""
    a = np.asarray(a, dtype=np.float64).reshape(-1, 7)
    b = np.asarray(b, dtype=np.float64).reshape(-1, 7)
    inter = bev_intersection(a, b)
    area_a, area_b = a[:, 3] * a[:, 4], b[:, 3] * b[:, 4]
    union = area_a[:, None] + area_b[None, :] - inter
    return np.where(union > 0, inter / np.maximum(union, 1e-12), 0.0)


def iou_3d(a: NDArray[np.floating], b: NDArray[np.floating]) -> FloatArray:
    """Pairwise 3D IoU of yaw-only 7-DoF gravity-centre boxes."""
    a = np.asarray(a, dtype=np.float64).reshape(-1, 7)
    b = np.asarray(b, dtype=np.float64).reshape(-1, 7)
    inter_bev = bev_intersection(a, b)
    za0, za1 = a[:, 2] - a[:, 5] / 2, a[:, 2] + a[:, 5] / 2
    zb0, zb1 = b[:, 2] - b[:, 5] / 2, b[:, 2] + b[:, 5] / 2
    dz = np.clip(np.minimum(za1[:, None], zb1[None]) - np.maximum(za0[:, None], zb0[None]), 0, None)
    inter = inter_bev * dz
    vol_a, vol_b = a[:, 3] * a[:, 4] * a[:, 5], b[:, 3] * b[:, 4] * b[:, 5]
    union = vol_a[:, None] + vol_b[None, :] - inter
    return np.where(union > 0, inter / np.maximum(union, 1e-12), 0.0)
