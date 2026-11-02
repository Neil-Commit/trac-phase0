"""Pose geometry and validation tests (synthetic; no webcam)."""

import math

import numpy as np
import pytest

from trac.config import PoseConfig
from trac.pose import (compute_pose, horizontal_angle_deg, planar_distance, validate_pose_values)

LIMITS = PoseConfig(min_distance_m=0.10, max_distance_m=6.0, max_reprojection_error_px=3.0)


def test_angle_convention_positive_x_is_right():
    assert horizontal_angle_deg(1.0, 1.0) == pytest.approx(45.0)
    assert horizontal_angle_deg(-1.0, 1.0) == pytest.approx(-45.0)
    assert horizontal_angle_deg(0.0, 2.0) == 0.0


def test_planar_distance_ignores_height():
    assert planar_distance(3.0, 4.0) == 5.0


def test_valid_values_accepted():
    assert validate_pose_values(0.2, 0.0, 1.5, 0.5, LIMITS) is None


@pytest.mark.parametrize("x,y,z,err", [
    (float("nan"), 0, 1.5, 0.5), (0, float("inf"), 1.5, 0.5),
    (0, 0, float("nan"), 0.5), (0, 0, 1.5, float("nan")), (0, 0, 1.5, float("inf")),
])
def test_non_finite_rejected(x, y, z, err):
    assert validate_pose_values(x, y, z, err, LIMITS)


@pytest.mark.parametrize("z", [0.0, -0.5])
def test_non_positive_z_rejected(z):
    assert validate_pose_values(0.0, 0.0, z, 0.5, LIMITS)


@pytest.mark.parametrize("x,z", [(0.0, 0.05), (0.0, 7.0), (5.0, 5.0)])
def test_unreasonable_distance_rejected(x, z):
    assert validate_pose_values(x, 0.0, z, 0.5, LIMITS)


def test_excessive_reprojection_error_rejected():
    assert validate_pose_values(0.0, 0.0, 1.5, 3.5, LIMITS)
    assert validate_pose_values(0.0, 0.0, 1.5, 3.0, LIMITS) is None  # boundary inclusive


# ----------------------------------------------- real solvePnP on projected corners
cv2 = pytest.importorskip("cv2")
from synthetic import make_calibration, project_corners  # noqa: E402

SIZE_M = 0.10


@pytest.mark.parametrize("x,z,yaw", [(0.0, 1.0, 20.0), (0.4, 1.5, -25.0), (-0.6, 2.5, 35.0), (0.1, 0.8, 10.0)])
def test_compute_pose_recovers_known_position(x, z, yaw):
    calib = make_calibration()
    corners = project_corners(calib, SIZE_M, x, z, yaw, noise_px=0.2)
    res = compute_pose(corners, 0, SIZE_M, calib.camera_matrix, calib.dist_coeffs, LIMITS)
    assert res.pose is not None, res.rejection_reason
    assert res.pose.x_m == pytest.approx(x, abs=0.03)
    assert res.pose.z_m == pytest.approx(z, abs=0.08)
    assert res.pose.angle_deg == pytest.approx(math.degrees(math.atan2(x, z)), abs=1.5)
    assert res.pose.reprojection_error_px < 1.0


def test_compute_pose_rejects_out_of_range_tag():
    calib = make_calibration()
    corners = project_corners(calib, SIZE_M, 0.0, 12.0, 10.0, noise_px=0.1)
    res = compute_pose(corners, 0, SIZE_M, calib.camera_matrix, calib.dist_coeffs, LIMITS)
    assert res.pose is None and "maximum" in res.rejection_reason


def test_compute_pose_rejects_non_finite_corners():
    calib = make_calibration()
    corners = project_corners(calib, SIZE_M, 0.0, 1.0)
    corners[2, 0] = np.nan
    res = compute_pose(corners, 0, SIZE_M, calib.camera_matrix, calib.dist_coeffs, LIMITS)
    assert res.pose is None


def test_compute_pose_rejects_geometrically_inconsistent_corners():
    calib = make_calibration()
    corners = project_corners(calib, SIZE_M, 0.0, 1.5, 0.0, noise_px=0.2)
    corners[0] += [18.0, -14.0]  # drag one corner far from any square's projection
    strict = PoseConfig(0.10, 6.0, 1.0)
    res = compute_pose(corners, 0, SIZE_M, calib.camera_matrix, calib.dist_coeffs, strict)
    assert res.pose is None and "Reprojection" in res.rejection_reason


def test_noise_free_degenerate_ippe_case_is_never_trusted():
    """Perfect noise-free pure-yaw corners make OpenCV's IPPE return a flipped rotation; the
    reprojection check must reject it rather than let a wrong pose through."""
    calib = make_calibration()
    corners = project_corners(calib, SIZE_M, 0.0, 1.0, 20.0, noise_px=0.0)
    res = compute_pose(corners, 0, SIZE_M, calib.camera_matrix, calib.dist_coeffs, LIMITS)
    assert res.pose is None or res.pose.reprojection_error_px <= LIMITS.max_reprojection_error_px
