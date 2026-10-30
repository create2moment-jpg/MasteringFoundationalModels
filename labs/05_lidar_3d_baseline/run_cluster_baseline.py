"""Lab 05 (CPU path): evaluate a classical LiDAR baseline on synthetic scenes.

Usage:
    python labs/05_lidar_3d_baseline/run_cluster_baseline.py --scenes 20
    python labs/05_lidar_3d_baseline/run_cluster_baseline.py --scenes 20 --save-bench benchmarks/results

Prints AP under several matching criteria and range bins, then a latency
record. Synthetic numbers are for understanding the metrics only; never
report them as detector performance.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from model_lab.baselines.clustering import ClusterConfig, detect
from model_lab.benchmarking.harness import BenchmarkConfig, run_benchmark
from model_lab.data.synthetic import make_scene
from model_lab.evaluation.ap import Criterion, FrameDetections, Interp, evaluate_ap, range_filter


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--scenes", type=int, default=20)
    ap.add_argument("--seed", type=int, default=100)
    ap.add_argument("--save-bench", type=Path, default=None)
    args = ap.parse_args()

    cfg = ClusterConfig()
    scenes = [make_scene(seed=args.seed + i) for i in range(args.scenes)]
    frames: list[FrameDetections] = []
    car_frames: list[FrameDetections] = []
    for s in scenes:
        d = detect(s.points, cfg)
        frames.append(FrameDetections(d.boxes, d.scores, s.boxes))
        # crude size rule: a "car" is anything longer than 2.5 m
        is_car = d.boxes[:, 3] > 2.5
        car_frames.append(
            FrameDetections(d.boxes[is_car], d.scores[is_car], s.boxes[s.labels == 0])
        )

    n_gt = sum(len(f.gt_boxes) for f in frames)
    n_pred = sum(len(f.pred_scores) for f in frames)
    print(f"{args.scenes} synthetic scenes, {n_gt} GT objects, {n_pred} detections\n")

    rows = [
        ("BEV IoU >= 0.7", Criterion.IOU_BEV, 0.7),
        ("BEV IoU >= 0.5", Criterion.IOU_BEV, 0.5),
        ("BEV IoU >= 0.25", Criterion.IOU_BEV, 0.25),
        ("3D IoU >= 0.5", Criterion.IOU_3D, 0.5),
        ("centre dist <= 1 m", Criterion.CENTER_DIST, 1.0),
        ("centre dist <= 2 m", Criterion.CENTER_DIST, 2.0),
        ("centre dist <= 4 m", Criterion.CENTER_DIST, 4.0),
    ]
    bins = [(0, 20), (20, 40), (40, 80)]
    header = f"{'criterion':<20} {'all-class AP':>12} {'car AP':>8} " + " ".join(
        f"{f'{a}-{b} m':>9}" for a, b in bins
    )
    print(header)
    print("-" * len(header))
    for name, crit, thr in rows:
        all_ap = evaluate_ap(frames, crit, thr, Interp.COCO_101).ap
        car_ap = evaluate_ap(car_frames, crit, thr, Interp.COCO_101).ap
        by_range = [
            evaluate_ap([range_filter(f, a, b) for f in frames], crit, thr).ap for a, b in bins
        ]
        print(
            f"{name:<20} {all_ap:>12.3f} {car_ap:>8.3f} " + " ".join(f"{v:>9.3f}" for v in by_range)
        )

    print("\nSame curve, different interpolation (all-class, centre dist <= 2 m):")
    for interp in Interp:
        print(
            f"  {interp.value:<16} {evaluate_ap(frames, Criterion.CENTER_DIST, 2.0, interp).ap:.3f}"
        )

    bench = run_benchmark(
        BenchmarkConfig(
            model_id="cluster-baseline",
            input_desc=f"synthetic 32-beam sweep ~{len(scenes[0].points)} pts",
            runtime="numpy",
            warmup=2,
            iters=min(10, len(scenes)),
        ),
        make_input=lambda i: scenes[i % len(scenes)].points,
        stages={"detect": lambda p: detect(p, cfg)},
    )
    e2e = bench.end_to_end
    print(f"\nlatency (CPU, numpy): p50 {e2e.p50_ms:.1f} ms, p90 {e2e.p90_ms:.1f} ms")
    if args.save_bench:
        print(f"benchmark record: {bench.save(args.save_bench)}")


if __name__ == "__main__":
    main()
