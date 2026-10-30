import numpy as np
import pytest

from model_lab.evaluation.ap import (
    Criterion,
    FrameDetections,
    Interp,
    average_precision,
    evaluate_ap,
    match_frame,
    range_filter,
)

GT = np.array(
    [[10, 0, 0, 4, 2, 1.5, 0], [20, 5, 0, 4, 2, 1.5, 0], [60, 0, 0, 4, 2, 1.5, 0]], dtype=float
)


def test_perfect_detections_give_ap_one_for_all_interpolations() -> None:
    f = FrameDetections(GT.copy(), np.array([0.9, 0.8, 0.7]), GT)
    for interp in Interp:
        assert evaluate_ap([f], Criterion.IOU_BEV, 0.7, interp).ap == pytest.approx(1.0)


def test_no_predictions_gives_zero() -> None:
    f = FrameDetections(np.zeros((0, 7)), np.zeros(0), GT)
    assert evaluate_ap([f], Criterion.IOU_3D, 0.5).ap == 0.0


def test_duplicate_detection_is_false_positive() -> None:
    preds = np.vstack([GT[0], GT[0]])
    _, tp = match_frame(
        FrameDetections(preds, np.array([0.9, 0.8]), GT[:1]), Criterion.IOU_BEV, 0.5
    )
    assert tp.tolist() == [True, False]


def test_center_distance_criterion() -> None:
    shifted = GT.copy()
    shifted[:, 0] += np.array([0.3, 1.5, 3.0])  # within 0.5 m, 2 m, 4 m respectively
    f = FrameDetections(shifted, np.array([0.9, 0.8, 0.7]), GT)
    assert match_frame(f, Criterion.CENTER_DIST, 0.5)[1].tolist() == [True, False, False]
    assert match_frame(f, Criterion.CENTER_DIST, 2.0)[1].tolist() == [True, True, False]
    assert match_frame(f, Criterion.CENTER_DIST, 4.0)[1].tolist() == [True, True, True]


def test_voc_all_points_on_handmade_curve() -> None:
    # TP, FP, TP over 2 GT -> precision [1, .5, .667], recall [.5, .5, 1]
    p = np.array([1.0, 0.5, 2 / 3])
    r = np.array([0.5, 0.5, 1.0])
    assert average_precision(p, r, Interp.VOC_ALL) == pytest.approx(0.5 * 1.0 + 0.5 * 2 / 3)


def test_high_confidence_false_positive_hurts_more_than_low() -> None:
    fp = np.array([[40, -10, 0, 4, 2, 1.5, 0]])
    preds = np.vstack([GT, fp])
    high = FrameDetections(preds, np.array([0.8, 0.7, 0.6, 0.95]), GT)
    low = FrameDetections(preds, np.array([0.8, 0.7, 0.6, 0.05]), GT)
    assert (
        evaluate_ap([high], Criterion.IOU_BEV, 0.5).ap
        < evaluate_ap([low], Criterion.IOU_BEV, 0.5).ap
    )


def test_range_filter() -> None:
    f = FrameDetections(GT.copy(), np.array([0.9, 0.8, 0.7]), GT)
    near = range_filter(f, 0, 30)
    assert len(near.gt_boxes) == 2 and len(near.pred_boxes) == 2
    assert len(range_filter(f, 50, 80).gt_boxes) == 1
