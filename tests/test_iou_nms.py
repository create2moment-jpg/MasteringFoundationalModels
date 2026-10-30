import numpy as np
import pytest

from model_lab.evaluation.iou import iou_2d, iou_3d, iou_bev
from model_lab.evaluation.nms import circle_nms, nms_2d, nms_bev


def test_iou_2d_known_values() -> None:
    a = np.array([[0, 0, 2, 2]])
    b = np.array([[1, 1, 3, 3], [0, 0, 2, 2], [5, 5, 6, 6]])
    np.testing.assert_allclose(iou_2d(a, b)[0], [1 / 7, 1.0, 0.0])


def test_iou_bev_rotation_invariance_and_known_overlap() -> None:
    a = np.array([[0, 0, 0, 2.0, 2.0, 1.0, 0.0]])
    # same square rotated 90 deg: identical footprint
    assert iou_bev(a, np.array([[0, 0, 0, 2.0, 2.0, 1.0, np.pi / 2]]))[0, 0] == pytest.approx(1.0)
    # square rotated 45 deg: intersection is a regular octagon of area 8(sqrt2-1)
    inter = 8 * (np.sqrt(2) - 1)
    expected = inter / (8 - inter)
    assert iou_bev(a, np.array([[0, 0, 0, 2.0, 2.0, 1.0, np.pi / 4]]))[0, 0] == pytest.approx(
        expected
    )
    # half shift along x
    assert iou_bev(a, np.array([[1.0, 0, 0, 2.0, 2.0, 1.0, 0.0]]))[0, 0] == pytest.approx(1 / 3)


def test_iou_3d_accounts_for_height() -> None:
    a = np.array([[0, 0, 0, 2.0, 2.0, 2.0, 0.0]])
    b = np.array([[0, 0, 1.0, 2.0, 2.0, 2.0, 0.0]])  # half vertical overlap
    assert iou_3d(a, b)[0, 0] == pytest.approx(1 / 3)
    assert iou_bev(a, b)[0, 0] == pytest.approx(1.0)


def test_nms_2d_suppresses_overlaps_keeps_order() -> None:
    boxes = np.array([[0, 0, 10, 10], [1, 1, 11, 11], [20, 20, 30, 30]])
    scores = np.array([0.6, 0.9, 0.5])
    assert nms_2d(boxes, scores, 0.5).tolist() == [1, 2]


def test_nms_bev_and_circle() -> None:
    boxes = np.array(
        [[0, 0, 0, 4, 2, 1.5, 0.0], [0.2, 0, 0, 4, 2, 1.5, 0.05], [10, 0, 0, 4, 2, 1.5, 0]]
    )
    scores = np.array([0.8, 0.7, 0.9])
    assert nms_bev(boxes, scores, 0.1).tolist() == [2, 0]
    assert circle_nms(boxes, scores, radius=1.0).tolist() == [2, 0]
