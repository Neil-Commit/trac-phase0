import json

import numpy as np
import pytest

from trac.calibration import (CalibrationError, CameraCalibration, calibrate_from_points, calibration_problem,
                              chessboard_object_points, load_calibration, save_calibration, try_load_calibration)
from synthetic import make_calibration

SIZE = (1280, 720)


def test_good_calibration_is_usable():
    assert calibration_problem(make_calibration(), SIZE, 1.5) is None


def test_missing_calibration_reported():
    assert calibration_problem(None, SIZE, 1.5)


def test_resolution_mismatch_rejected():
    assert "1280x720" in calibration_problem(make_calibration(), (640, 480), 1.5)


def test_high_error_rejected():
    assert calibration_problem(make_calibration(rms=2.0), SIZE, 1.5)


@pytest.mark.parametrize("mutate", [
    lambda c: c.camera_matrix.__setitem__((0, 0), np.nan),
    lambda c: c.camera_matrix.__setitem__((0, 0), -5.0),
    lambda c: c.camera_matrix.__setitem__((0, 2), 5000.0),
    lambda c: c.dist_coeffs.__setitem__((0, 1), np.inf),
])
def test_malformed_calibration_rejected(mutate):
    calib = make_calibration()
    mutate(calib)
    assert calibration_problem(calib, SIZE, 1.5)


def test_nan_error_rejected():
    assert calibration_problem(make_calibration(rms=float("nan")), SIZE, 1.5)


def test_save_load_round_trip(tmp_path):
    original = make_calibration()
    path = tmp_path / "sub" / "cal.json"
    save_calibration(path, original)
    loaded = load_calibration(path)
    assert np.allclose(loaded.camera_matrix, original.camera_matrix)
    assert np.allclose(loaded.dist_coeffs, original.dist_coeffs)
    assert (loaded.image_width, loaded.image_height) == SIZE
    assert loaded.rms_error_px == original.rms_error_px
    saved = json.loads(path.read_text())
    assert {"camera_matrix", "dist_coeffs", "image_width", "image_height", "rms_error_px"} <= set(saved)


def test_missing_and_corrupt_files(tmp_path):
    calib, note = try_load_calibration(tmp_path / "nope.json")
    assert calib is None and "not found" in note
    bad = tmp_path / "bad.json"
    bad.write_text("{not json")
    assert try_load_calibration(bad)[0] is None
    bad.write_text(json.dumps({"camera_matrix": [[1]]}))
    with pytest.raises(CalibrationError):
        load_calibration(bad)


cv2 = pytest.importorskip("cv2")


def test_calibrate_from_synthetic_chessboard_views():
    truth = make_calibration()
    rng = np.random.default_rng(3)
    obj = chessboard_object_points(9, 6, 0.025)
    centre = obj.mean(axis=0)
    obj_sets, img_sets = [], []
    for _ in range(18):
        rvec = rng.uniform(-0.5, 0.5, 3)
        tvec = np.array([rng.uniform(-0.05, 0.05), rng.uniform(-0.03, 0.03), rng.uniform(0.35, 0.7)])
        R = cv2.Rodrigues(rvec)[0]
        t = tvec - R @ centre  # keep the board near the optical axis
        pts, _ = cv2.projectPoints(obj, cv2.Rodrigues(R)[0], t, truth.camera_matrix, truth.dist_coeffs)
        obj_sets.append(obj)
        img_sets.append((pts.reshape(-1, 2) + rng.normal(0, 0.15, (len(obj), 2))).astype(np.float32))
    calib = calibrate_from_points(obj_sets, img_sets, SIZE)
    assert calib.fx == pytest.approx(800.0, rel=0.03)
    assert calib.fy == pytest.approx(800.0, rel=0.03)
    assert calib.rms_error_px < 0.5
    assert calibration_problem(calib, SIZE, 1.5) is None


def test_calibrate_needs_enough_views():
    with pytest.raises(CalibrationError):
        calibrate_from_points([], [], SIZE)
