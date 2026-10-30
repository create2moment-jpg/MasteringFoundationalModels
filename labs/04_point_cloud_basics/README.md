# Lab 04: point-cloud basics (CPU, no downloads)

**Objective:** build intuition for the properties of automotive LiDAR that every 3D model
depends on: frames and box conventions, range fall-off, occlusion, voxel/pillar occupancy,
and camera projection.

**Runs on:** any CPU in under 1 s. **Executed:** yes (build container, 2026-10-30).

```bash
python labs/04_point_cloud_basics/explore_scene.py --seed 3 --out outputs/lab04
```

## What you get (seed 3, as executed)

- About 24.5k points (`x, y, z, intensity`) from a simulated 32-beam spinning LiDAR. The median range is about 7 m, because most returns hit nearby ground.
- A per-object table of points by range. In seed 3, a car at 10 m gets 259 points, a car at 41 m gets 18, and **a car at 33 m gets 0** (fully occluded).
- Pillar grid 432x496 (KITTI PointPillars config) at about 2.4% occupancy. Voxel grid 1024x1024x40 at about 0.04% occupancy. **This sparsity is why sparse convolution exists.**
- `bev_density.pgm`, `bev_max_height.pgm`, `camera_inverse_depth.pgm` in `--out`. About 21% of points project into the front camera.

## Explain these (self-assessment)

1. Why does point count per object fall faster than 1/r for distant cars? (Hint: both angular dimensions shrink, and so does incidence.)
2. A detector's labels include the 0-point car. What should evaluation do with it? (Compare the KITTI "difficulty" filter with nuScenes's "≥1 LiDAR or radar point" filter.)
3. Read `geometry.py`. Which two conversions does a KITTI label need before it matches an OpenPCDet box, and which one does an mmdetection3d box need?
4. Why is `coords_zyx` in `(z, y, x)` order? Which library expects that?
5. Pillars at 0.16 m vs voxels at 0.1x0.1x0.2 m: what do you gain and lose?

## Why it works / when it fails

The simulator gets geometry right (ray casting, occlusion, angular resolution). It has no
intensity physics, motion distortion, rain, multi-return or rolling sweeps, so **do not tune
models on it**. Move to KITTI or nuScenes mini for anything quantitative.

**Stretch:** add a 0.1 s ego-motion during the sweep (rotate the azimuth by time) and visualise the
"rolling shutter" seam in BEV. This is the effect that motion compensation removes.

**Cleanup:** `rm -rf outputs/lab04`
