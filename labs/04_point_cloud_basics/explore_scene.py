"""Lab 04: point-cloud geometry on a synthetic spinning-LiDAR scene (CPU, no downloads).

Usage:
    python labs/04_point_cloud_basics/explore_scene.py --seed 3 --out outputs/lab04

Writes BEV density / height images (PGM, viewable in most image viewers) and a
camera-projection depth image to --out (git-ignored), and prints the numbers
the README asks you to explain.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from model_lab.data.geometry import lidar_points_to_image, points_in_boxes
from model_lab.data.synthetic import CLASS_NAMES, make_scene
from model_lab.data.voxelize import VoxelGrid, bev_occupancy, voxelize


def write_pgm(path: Path, img: NDArray[np.floating]) -> None:
    """8-bit PGM (P5) writer; avoids an imaging dependency."""
    a = np.asarray(img, dtype=np.float64)
    lo, hi = np.nanmin(a), np.nanmax(a)
    u8 = np.zeros(a.shape, np.uint8) if hi <= lo else ((a - lo) / (hi - lo) * 255).astype(np.uint8)
    path.write_bytes(f"P5 {a.shape[1]} {a.shape[0]} 255\n".encode() + u8.tobytes())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--out", type=Path, default=Path("outputs/lab04"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    s = make_scene(seed=args.seed)
    pts = s.points
    rng = np.hypot(pts[:, 0], pts[:, 1])
    print(f"points: {pts.shape}  (x, y, z, intensity), dtype {pts.dtype}")
    print(f"range: min {rng.min():.1f} m, median {np.median(rng):.1f} m, max {rng.max():.1f} m")

    # 1. Range fall-off: points per object vs distance
    inside = points_in_boxes(pts, s.boxes)
    print("\nobjects (lab box convention [x, y, z, l, w, h, yaw]):")
    print(f"{'#':>2} {'class':<10} {'range':>6} {'yaw':>6} {'points':>7}")
    for k, (b, lab) in enumerate(zip(s.boxes, s.labels, strict=True)):
        r = np.hypot(b[0], b[1])
        print(f"{k:>2} {CLASS_NAMES[lab]:<10} {r:6.1f} {b[6]:6.2f} {int(inside[:, k].sum()):>7}")

    # 2. Voxelisation: KITTI PointPillars grid vs nuScenes-like voxel grid
    pillars = VoxelGrid((0.16, 0.16, 4.0), (0.0, -39.68, -3.0, 69.12, 39.68, 1.0))
    voxels = VoxelGrid((0.1, 0.1, 0.2), (-51.2, -51.2, -5.0, 51.2, 51.2, 3.0))
    for name, grid, cap in (("pillars", pillars, 32), ("voxels", voxels, 10)):
        v = voxelize(pts, grid, max_points_per_voxel=cap, max_voxels=200_000)
        n_cells = int(np.prod(grid.grid_size))
        full = int((v.num_points == cap).sum())
        print(
            f"\n{name}: grid {grid.grid_size} = {n_cells:,} cells, non-empty {len(v.num_points):,} "
            f"({100 * len(v.num_points) / n_cells:.3f}% occupancy), at capacity {full}"
        )

    # 3. BEV rasters
    density, height = bev_occupancy(pts, pillars)
    write_pgm(args.out / "bev_density.pgm", np.log1p(density)[::-1, ::-1])
    write_pgm(args.out / "bev_max_height.pgm", height[::-1, ::-1])

    # 4. Projection into the front camera
    uv, depth, valid = lidar_points_to_image(pts, s.t_cam_lidar, s.intrinsics, s.image_hw)
    h, w = s.image_hw
    depth_img = np.zeros((h, w))
    u, vv = uv[valid, 0].astype(int), uv[valid, 1].astype(int)
    order = np.argsort(-depth[valid])  # far first so near points overwrite
    depth_img[vv[order], u[order]] = 1.0 / depth[valid][order]
    write_pgm(args.out / "camera_inverse_depth.pgm", depth_img)
    print(
        f"\nprojected into front camera: {valid.sum():,} / {len(pts):,} points ({valid.mean():.1%})"
    )
    print(f"wrote {sorted(p.name for p in args.out.iterdir())} to {args.out}/")


if __name__ == "__main__":
    main()
