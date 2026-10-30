import numpy as np
import pytest

from model_lab.data.geometry import (
    R_CAM_FROM_LIDAR_AXES,
    bottom_to_gravity_center,
    box_corners_3d,
    gravity_to_bottom_center,
    invert_transform,
    kitti_camera_boxes_to_lidar,
    lidar_points_to_image,
    limit_period,
    make_transform,
    points_in_boxes,
    project_to_image,
    rot_z,
    transform_points,
)


def test_limit_period_wraps_into_minus_pi_pi() -> None:
    a = np.array([0.0, np.pi, -np.pi, 3 * np.pi / 2, -3 * np.pi / 2, 7.0])
    w = limit_period(a)
    assert np.all(w >= -np.pi) and np.all(w < np.pi)
    np.testing.assert_allclose(np.cos(w), np.cos(a), atol=1e-12)
    np.testing.assert_allclose(np.sin(w), np.sin(a), atol=1e-12)


def test_transform_roundtrip_and_passthrough_columns() -> None:
    rng = np.random.default_rng(0)
    t = make_transform(rot_z(0.7) @ np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]]), [1.0, -2.0, 0.5])
    pts = np.column_stack([rng.normal(size=(50, 3)), rng.uniform(size=50)])
    back = transform_points(invert_transform(t), transform_points(t, pts))
    np.testing.assert_allclose(back, pts, atol=1e-12)
    np.testing.assert_array_equal(transform_points(t, pts)[:, 3], pts[:, 3])


def test_box_corners_axis_aligned_and_yaw_convention() -> None:
    box = np.array([[10.0, 0.0, 0.0, 4.0, 2.0, 1.5, 0.0]])
    c = box_corners_3d(box)[0]
    assert c.shape == (8, 3)
    np.testing.assert_allclose(c[0], [12.0, 1.0, -0.75])  # front-left-bottom
    np.testing.assert_allclose(c[:4, 2], -0.75)
    np.testing.assert_allclose(c[4:, 2], 0.75)
    # yaw = +90 deg: heading points along +y (counter-clockwise from +x)
    c90 = box_corners_3d(np.array([[0, 0, 0, 4.0, 2.0, 1.0, np.pi / 2]]))[0]
    front_center = c90[[0, 3, 4, 7]].mean(axis=0)
    np.testing.assert_allclose(front_center, [0.0, 2.0, 0.0], atol=1e-12)


def test_points_in_boxes_respects_rotation() -> None:
    box = np.array([[0, 0, 0, 4.0, 1.0, 2.0, np.pi / 2]])  # long axis along y
    pts = np.array([[0, 1.9, 0], [1.9, 0, 0], [0, 0, 0.99], [0, 0, 1.01]])
    np.testing.assert_array_equal(points_in_boxes(pts, box)[:, 0], [True, False, True, False])


def test_bottom_gravity_roundtrip() -> None:
    b = np.array([[1, 2, -1.0, 4, 2, 1.6, 0.3]])
    g = bottom_to_gravity_center(b)
    assert g[0, 2] == pytest.approx(-0.2)
    np.testing.assert_allclose(gravity_to_bottom_center(g), b)


def test_kitti_camera_to_lidar_matches_standard_formula() -> None:
    # Ideal KITTI-like extrinsics: pure axis permutation plus a small offset.
    t_cam_lidar = make_transform(R_CAM_FROM_LIDAR_AXES, [0.0, -0.08, -0.27])
    t_lidar_cam = invert_transform(t_cam_lidar)
    ry = 0.4
    kitti = np.array([[2.0, 1.6, 20.0, 1.5, 1.6, 3.9, ry]])  # x, y(bottom), z, h, w, l, ry
    out = kitti_camera_boxes_to_lidar(kitti, t_lidar_cam)[0]
    center_cam_gravity = np.array([[2.0, 1.6 - 0.75, 20.0]])
    np.testing.assert_allclose(out[:3], transform_points(t_lidar_cam, center_cam_gravity)[0])
    np.testing.assert_allclose(out[3:6], [3.9, 1.6, 1.5])  # l, w, h
    assert out[6] == pytest.approx(limit_period(np.array(-ry - np.pi / 2)))


def test_projection_behind_camera_is_invalid() -> None:
    k = np.array([[100.0, 0, 50], [0, 100.0, 40], [0, 0, 1]])
    uv, depth, valid = project_to_image(np.array([[0, 0, 10.0], [1, 1, -5.0]]), k)
    np.testing.assert_allclose(uv[0], [50, 40])
    assert valid.tolist() == [True, False]
    assert np.isnan(uv[1]).all()
    assert depth[1] == -5.0


def test_lidar_point_ahead_projects_to_principal_point() -> None:
    t = make_transform(R_CAM_FROM_LIDAR_AXES, [0, 0, 0])
    k = np.array([[700.0, 0, 640], [0, 700.0, 360], [0, 0, 1]])
    pts = np.array([[20.0, 0, 0], [-20.0, 0, 0], [20.0, 5.0, 0]])  # ahead, behind, ahead-left
    uv, _, valid = lidar_points_to_image(pts, t, k, (720, 1280))
    np.testing.assert_allclose(uv[0], [640, 360])
    assert valid.tolist() == [True, False, True]
    assert uv[2, 0] < 640  # +y (left) in LiDAR -> smaller u in image
