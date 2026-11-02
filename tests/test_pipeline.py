"""End-to-end pipeline tests on rendered synthetic frames (no webcam)."""

import numpy as np
import pytest

pytest.importorskip("cv2")

from trac.config import AppConfig, TagConfig
from trac.navigation import NavState
from trac.pipeline import TracPipeline, split_detections
from trac.detector import TagDetection
from synthetic import make_calibration, render_scene

SIZE_M = 0.10
CFG = AppConfig(tag=TagConfig(assigned_id=0, size_mm=100.0))
BLANK = np.full((720, 1280, 3), 255, dtype=np.uint8)


def run(pipe, frame, n=6):
    r = None
    for _ in range(n):
        r = pipe.process(frame)
    return r


def test_split_detections_only_assigned_id_selected():
    dets = [TagDetection(i, np.zeros((4, 2), np.float32)) for i in (3, 0, 7)]
    assigned, others = split_detections(dets, 0)
    assert [d.tag_id for d in assigned] == [0] and [d.tag_id for d in others] == [3, 7]


def test_assigned_tag_far_away_is_followed():
    calib = make_calibration()
    frame = render_scene(calib, SIZE_M, [(0, 0.0, 2.0, 15.0)])
    r = run(TracPipeline(CFG, calib), frame)
    assert r.calibrated and r.pose is not None
    assert r.pose.z_m == pytest.approx(2.0, abs=0.15)
    assert r.decision.state == NavState.FOLLOW_FORWARD and r.decision.command.linear_mps > 0


def test_target_on_right_turns_right():
    calib = make_calibration()
    frame = render_scene(calib, SIZE_M, [(0, 0.8, 2.0, 15.0)])
    r = run(TracPipeline(CFG, calib), frame)
    assert r.pose.x_m > 0 and r.pose.angle_deg > 0
    assert r.decision.state == NavState.FOLLOW_RIGHT and r.decision.command.angular_rps < 0


def test_wrong_tag_does_not_affect_navigation():
    calib = make_calibration()
    frame = render_scene(calib, SIZE_M, [(5, 0.0, 2.0, 10.0)])
    r = run(TracPipeline(CFG, calib), frame)
    assert r.other_ids == [5] and r.assigned is None and r.pose is None
    assert r.decision.state == NavState.WRONG_TAG_STOP and r.decision.command.is_zero


def test_other_tags_are_ignored_when_assigned_tag_present():
    calib = make_calibration()
    alone = run(TracPipeline(CFG, calib), render_scene(calib, SIZE_M, [(0, 0.0, 2.0, 10.0)]))
    with_others = run(TracPipeline(CFG, calib),
                      render_scene(calib, SIZE_M, [(0, 0.0, 2.0, 10.0), (4, -0.9, 1.2, 0.0), (9, 0.9, 1.8, 0.0)]))
    assert with_others.other_ids == [4, 9]
    assert with_others.decision.state == alone.decision.state
    assert with_others.decision.command == alone.decision.command


def test_no_tag_stops():
    r = TracPipeline(CFG, make_calibration()).process(BLANK)
    assert r.decision.state == NavState.NO_TARGET_STOP and r.decision.command.is_zero


def test_uncalibrated_shows_tags_but_disables_metrics_and_movement():
    calib = make_calibration()
    frame = render_scene(calib, SIZE_M, [(0, 0.0, 2.0, 15.0)])
    r = run(TracPipeline(CFG, None, "Calibration file not found: x"), frame)
    assert [d.tag_id for d in r.detections] == [0]          # detection and ID display still work
    assert not r.calibrated and r.pose is None
    assert r.decision.state == NavState.UNCALIBRATED_STOP and r.decision.command.is_zero
    assert "not found" in r.calibration_note


def test_resolution_mismatch_is_uncalibrated():
    calib = make_calibration(width=640, height=480, f=500.0)
    frame = render_scene(make_calibration(), SIZE_M, [(0, 0.0, 2.0, 15.0)])  # 1280x720 frame
    r = run(TracPipeline(CFG, calib), frame)
    assert r.decision.state == NavState.UNCALIBRATED_STOP and r.decision.command.is_zero


def test_tag_too_close_stops():
    calib = make_calibration()
    frame = render_scene(calib, SIZE_M, [(0, 0.0, 0.35, 15.0)])
    r = run(TracPipeline(CFG, calib), frame)
    assert r.decision.state == NavState.TOO_CLOSE_STOP and r.decision.command.is_zero


def test_duplicate_assigned_tags_are_ambiguous_and_stop():
    calib = make_calibration()
    frame = render_scene(calib, SIZE_M, [(0, -0.6, 2.0, 10.0), (0, 0.6, 2.0, 10.0)])
    r = run(TracPipeline(CFG, calib), frame)
    assert r.decision.state == NavState.POSE_INVALID_STOP and r.decision.command.is_zero


def test_losing_the_tag_stops_immediately():
    calib = make_calibration()
    pipe = TracPipeline(CFG, calib)
    assert run(pipe, render_scene(calib, SIZE_M, [(0, 0.0, 2.0, 15.0)])).decision.command.linear_mps > 0
    r = pipe.process(BLANK)
    assert r.decision.state == NavState.NO_TARGET_STOP and r.decision.command.is_zero


def test_no_camera_frame_stops():
    r = TracPipeline(CFG, make_calibration()).process_no_frame()
    assert r.decision.state == NavState.NO_TARGET_STOP and r.decision.command.is_zero
