import numpy as np

from model_lab.baselines.clustering import ClusterConfig, bev_connected_components, detect, fit_box
from model_lab.data.synthetic import make_scene
from model_lab.evaluation.ap import Criterion, FrameDetections, evaluate_ap


def test_connected_components_8_neighbourhood() -> None:
    xy = np.array([[0.1, 0.1], [0.4, 0.4], [5.0, 5.0], [0.7, 0.7]])  # diagonal chain + far point
    lab = bev_connected_components(xy, cell=0.3)
    assert lab[0] == lab[1] == lab[3] != lab[2]
    assert len(bev_connected_components(np.zeros((0, 2)), 0.3)) == 0


def test_fit_box_full_rectangle_recovers_box() -> None:
    gx, gy = np.meshgrid(np.linspace(-2, 2, 21), np.linspace(-0.9, 0.9, 10))
    xy = np.column_stack([gx.ravel(), gy.ravel()])
    yaw = 0.5
    rot = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
    pts = np.column_stack([xy @ rot.T + [10, 3], np.full(len(xy), -0.2)])
    box = fit_box(pts, ground_z=-1.73)
    np.testing.assert_allclose(box[:2], [10, 3], atol=1e-6)
    np.testing.assert_allclose(box[3:5], [4.0, 1.8], atol=1e-6)
    assert np.isclose(np.cos(2 * (box[6] - yaw)), 1.0)  # yaw up to pi ambiguity


def test_detect_end_to_end_beats_chance_on_synthetic() -> None:
    frames = []
    for seed in range(3):
        s = make_scene(seed=seed)
        d = detect(s.points, ClusterConfig())
        assert d.boxes.shape[1] == 7 and len(d.scores) == len(d.boxes)
        frames.append(FrameDetections(d.boxes, d.scores, s.boxes))
    assert evaluate_ap(frames, Criterion.CENTER_DIST, 2.0).ap > 0.4
