import numpy as np

from model_lab.data.geometry import lidar_points_to_image, points_in_boxes
from model_lab.data.synthetic import make_scene
from model_lab.data.voxelize import VoxelGrid, bev_occupancy, voxelize

KITTI_PILLARS = VoxelGrid((0.16, 0.16, 4.0), (0.0, -39.68, -3.0, 69.12, 39.68, 1.0))


def test_grid_size_matches_openpcdet_kitti_pointpillars() -> None:
    assert KITTI_PILLARS.grid_size == (432, 496, 1)


def test_voxelize_capacity_ordering_and_coords() -> None:
    grid = VoxelGrid((1.0, 1.0, 1.0), (0, 0, 0, 4, 4, 4))
    pts = np.array(
        [
            [0.5, 0.5, 0.5, 1],
            [0.6, 0.5, 0.5, 2],
            [0.7, 0.5, 0.5, 3],
            [3.5, 2.5, 1.5, 4],
            [9, 9, 9, 5],
        ],
        dtype=np.float32,
    )
    v = voxelize(pts, grid, max_points_per_voxel=2, max_voxels=10)
    assert v.features.shape == (2, 2, 4)
    assert v.num_points.tolist() == [2, 1]  # capacity 2: third point dropped; out-of-range dropped
    assert v.features[0, :, 3].tolist() == [1, 2]  # first-come order kept
    assert v.coords_zyx.tolist() == [[0, 0, 0], [1, 2, 3]]  # (z, y, x)


def test_voxelize_max_voxels_truncates() -> None:
    grid = VoxelGrid((1.0, 1.0, 1.0), (0, 0, 0, 10, 1, 1))
    pts = np.column_stack([np.arange(10) + 0.5, np.full(10, 0.5), np.full(10, 0.5)]).astype(
        np.float32
    )
    assert len(voxelize(pts, grid, max_voxels=3).num_points) == 3


def test_synthetic_scene_is_deterministic_and_plausible() -> None:
    a, b = make_scene(seed=7), make_scene(seed=7)
    np.testing.assert_array_equal(a.points, b.points)
    assert a.points.shape[1] == 4 and len(a.points) > 10000
    assert len(a.boxes) > 0
    ground = a.points[:, 2] < -1.6
    assert ground.mean() > 0.5  # most returns hit the ground
    inside = points_in_boxes(a.points, a.boxes * np.array([1, 1, 1, 1.05, 1.05, 1.05, 1]))
    hits_per_box = inside.sum(axis=0)
    assert (hits_per_box > 0).sum() >= len(a.boxes) // 2  # occlusion may hide a few
    # near objects get more points than far objects of the same class (range fall-off)
    rng = np.hypot(a.boxes[:, 0], a.boxes[:, 1])
    cars = a.labels == 0
    if cars.sum() >= 2:
        near, far = np.argmin(np.where(cars, rng, np.inf)), np.argmax(np.where(cars, rng, -np.inf))
        assert hits_per_box[near] >= hits_per_box[far]


def test_bev_occupancy_and_camera_projection_on_scene() -> None:
    s = make_scene(seed=1)
    density, height = bev_occupancy(s.points, KITTI_PILLARS)
    assert density.shape == (496, 432) and density.sum() > 0
    assert height.max() > -3.0
    _, depth, valid = lidar_points_to_image(s.points, s.t_cam_lidar, s.intrinsics, s.image_hw)
    assert 0 < valid.sum() < len(s.points)
    assert (depth[valid] > 0).all()
