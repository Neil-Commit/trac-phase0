"""Tag pose estimation and validation.

Camera coordinate convention (OpenCV): +x right, +y down, +z forward.
Horizontal angle = atan2(x, z) (positive = target to the right).
Planar distance  = sqrt(x^2 + z^2) (ignores height).

``validate_pose_values`` and the geometry helpers are pure Python (no OpenCV) so they can
be unit-tested with synthetic numbers. ``compute_pose`` wraps ``cv2.solvePnP``
(SOLVEPNP_IPPE_SQUARE) and rejects anything that fails validation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .config import PoseConfig


def horizontal_angle_deg(x: float, z: float) -> float:
    return math.degrees(math.atan2(x, z))


def planar_distance(x: float, z: float) -> float:
    return math.sqrt(x * x + z * z)


@dataclass(frozen=True)
class TagPose:
    """A pose that has already passed validation."""

    tag_id: int
    x_m: float
    y_m: float
    z_m: float
    distance_m: float
    angle_deg: float
    reprojection_error_px: float
    rvec: np.ndarray = field(repr=False, compare=False)
    tvec: np.ndarray = field(repr=False, compare=False)


@dataclass(frozen=True)
class PoseResult:
    pose: Optional[TagPose]
    rejection_reason: Optional[str] = None


def validate_pose_values(x: float, y: float, z: float, reprojection_error_px: float,
                         limits: PoseConfig) -> Optional[str]:
    """Return None if the values describe a usable pose, else a rejection reason."""
    if not all(math.isfinite(v) for v in (x, y, z, reprojection_error_px)):
        return "Pose contains non-finite values"
    if z <= 0:
        return f"Tag is not in front of the camera (z = {z:.2f} m)"
    dist = planar_distance(x, z)
    if dist < limits.min_distance_m:
        return f"Distance {dist:.2f} m below plausible minimum {limits.min_distance_m:.2f} m"
    if dist > limits.max_distance_m:
        return f"Distance {dist:.2f} m above plausible maximum {limits.max_distance_m:.2f} m"
    if reprojection_error_px > limits.max_reprojection_error_px:
        return (f"Reprojection error {reprojection_error_px:.1f} px exceeds limit "
                f"{limits.max_reprojection_error_px:.1f} px")
    return None


def tag_object_points(tag_size_m: float) -> np.ndarray:
    """3D corners of a square tag centred at the origin, in detector corner order."""
    h = tag_size_m / 2.0
    return np.array([[-h, h, 0], [h, h, 0], [h, -h, 0], [-h, -h, 0]], dtype=np.float32)


def compute_pose(corners: np.ndarray, tag_id: int, tag_size_m: float,
                 camera_matrix: np.ndarray, dist_coeffs: np.ndarray,
                 limits: PoseConfig) -> PoseResult:
    """Estimate and validate the pose of one tag from its 4 image corners."""
    import cv2  # local import keeps the pure helpers above importable without OpenCV

    image_points = np.asarray(corners, dtype=np.float32).reshape(4, 2)
    if not np.all(np.isfinite(image_points)):
        return PoseResult(None, "Tag corners contain non-finite values")
    object_points = tag_object_points(tag_size_m)

    try:
        ok, rvec, tvec = cv2.solvePnP(
            object_points, image_points.reshape(4, 1, 2), camera_matrix, dist_coeffs,
            flags=cv2.SOLVEPNP_IPPE_SQUARE,
        )
    except cv2.error as exc:
        return PoseResult(None, f"solvePnP error: {exc.msg.strip()[:80]}")
    if not ok:
        return PoseResult(None, "solvePnP failed")

    projected, _ = cv2.projectPoints(object_points, rvec, tvec, camera_matrix, dist_coeffs)
    residuals = projected.reshape(4, 2) - image_points
    reproj = float(np.sqrt(np.mean(np.sum(residuals ** 2, axis=1))))

    x, y, z = (float(v) for v in tvec.reshape(3))
    reason = validate_pose_values(x, y, z, reproj, limits)
    if reason is not None:
        return PoseResult(None, reason)

    return PoseResult(TagPose(
        tag_id=tag_id, x_m=x, y_m=y, z_m=z,
        distance_m=planar_distance(x, z), angle_deg=horizontal_angle_deg(x, z),
        reprojection_error_px=reproj, rvec=rvec, tvec=tvec,
    ))
