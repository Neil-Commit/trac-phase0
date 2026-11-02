"""Synthetic scenes for tests: project/render AprilTags with a known camera and pose."""

from __future__ import annotations

import math
from typing import Sequence, Tuple

import cv2
import numpy as np

from trac.calibration import CameraCalibration
from trac.pose import tag_object_points

W, H, F = 1280, 720, 800.0


def make_calibration(width: int = W, height: int = H, f: float = F, rms: float = 0.3) -> CameraCalibration:
    k = np.array([[f, 0, width / 2], [0, f, height / 2], [0, 0, 1]], dtype=np.float64)
    return CameraCalibration(k, np.zeros((1, 5)), width, height, rms, rms, 15, "test")


def tag_rt(x: float, z: float, yaw_deg: float = 0.0, y: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
    """Pose of a tag facing the camera, centre at (x, y, z) m, rotated yaw_deg about vertical.

    OpenCV's tag frame has +y up, so a camera-facing tag is a 180-degree rotation about x.
    """
    yaw = math.radians(yaw_deg)
    ry = np.array([[math.cos(yaw), 0, math.sin(yaw)], [0, 1, 0], [-math.sin(yaw), 0, math.cos(yaw)]])
    rx = np.diag([1.0, -1.0, -1.0])
    rvec, _ = cv2.Rodrigues(ry @ rx)
    return rvec, np.array([[x], [y], [z]], dtype=np.float64)


def project_corners(calib: CameraCalibration, size_m: float, x: float, z: float,
                    yaw_deg: float = 0.0, noise_px: float = 0.0, seed: int = 0) -> np.ndarray:
    """Project the tag corners. Real detections always carry sub-pixel noise; use noise_px > 0
    for pose-recovery tests (OpenCV's IPPE has a degenerate case for mathematically perfect,
    noise-free pure-yaw corners, which compute_pose correctly rejects via reprojection error)."""
    rvec, tvec = tag_rt(x, z, yaw_deg)
    pts, _ = cv2.projectPoints(tag_object_points(size_m), rvec, tvec, calib.camera_matrix, calib.dist_coeffs)
    pts = pts.reshape(4, 2)
    if noise_px > 0:
        pts = pts + np.random.default_rng(seed).normal(0.0, noise_px, pts.shape)
    return pts.astype(np.float32)


def render_scene(calib: CameraCalibration, size_m: float,
                 tags: Sequence[Tuple[int, float, float, float]]) -> np.ndarray:
    """Render tags given as (id, x_m, z_m, yaw_deg) into a BGR frame."""
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    side, margin = 160, 48
    frame = np.full((calib.image_height, calib.image_width), 255, dtype=np.uint8)
    src = np.float32([[margin, margin], [margin + side, margin],
                      [margin + side, margin + side], [margin, margin + side]])
    for tag_id, x, z, yaw in tags:
        canvas = np.full((side + 2 * margin,) * 2, 255, dtype=np.uint8)
        canvas[margin:margin + side, margin:margin + side] = cv2.aruco.generateImageMarker(dictionary, tag_id, side, 1)
        dst = project_corners(calib, size_m, x, z, yaw)
        warp = cv2.warpPerspective(canvas, cv2.getPerspectiveTransform(src, dst),
                                   (calib.image_width, calib.image_height), borderValue=255)
        frame = np.minimum(frame, warp)
    return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
