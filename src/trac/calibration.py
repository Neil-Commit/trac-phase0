"""Camera calibration storage, validation, and computation.

The saved JSON contains the camera matrix, distortion coefficients, the resolution the
calibration was made at, and calibration error. Calibration is only usable at the same
resolution it was captured at.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

FORMAT_VERSION = 1


class CalibrationError(RuntimeError):
    """Raised when calibration cannot be loaded or computed."""


@dataclass(frozen=True, eq=False)
class CameraCalibration:
    camera_matrix: np.ndarray            # 3x3
    dist_coeffs: np.ndarray              # 1xN
    image_width: int
    image_height: int
    rms_error_px: float                  # RMS reprojection error from cv2.calibrateCamera
    mean_view_error_px: Optional[float] = None
    num_views: int = 0
    created_utc: str = ""
    pattern: Dict[str, Any] = field(default_factory=dict)

    @property
    def fx(self) -> float:
        return float(self.camera_matrix[0, 0])

    @property
    def fy(self) -> float:
        return float(self.camera_matrix[1, 1])

    @property
    def cx(self) -> float:
        return float(self.camera_matrix[0, 2])

    @property
    def cy(self) -> float:
        return float(self.camera_matrix[1, 2])


def calibration_problem(calib: Optional[CameraCalibration], frame_size: Tuple[int, int],
                        max_rms_px: float) -> Optional[str]:
    """Return why this calibration cannot be used for a frame of ``frame_size`` (w, h), or None."""
    if calib is None:
        return "No calibration loaded"
    k, d = np.asarray(calib.camera_matrix, dtype=float), np.asarray(calib.dist_coeffs, dtype=float)
    if k.shape != (3, 3) or not np.all(np.isfinite(k)):
        return "Camera matrix is malformed or non-finite"
    if d.ndim != 2 or d.shape[0] != 1 or d.size == 0 or d.size > 14 or not np.all(np.isfinite(d)):
        return "Distortion coefficients are malformed or non-finite"
    if k[0, 0] <= 0 or k[1, 1] <= 0:
        return "Focal lengths must be positive"
    if not (0 < k[0, 2] < calib.image_width and 0 < k[1, 2] < calib.image_height):
        return "Principal point lies outside the calibration image"
    if not math.isfinite(calib.rms_error_px) or calib.rms_error_px < 0:
        return "Calibration error is missing or invalid"
    if calib.rms_error_px > max_rms_px:
        return f"Calibration error {calib.rms_error_px:.2f} px exceeds limit {max_rms_px:.2f} px"
    if (calib.image_width, calib.image_height) != tuple(frame_size):
        return (f"Calibrated at {calib.image_width}x{calib.image_height} "
                f"but camera delivers {frame_size[0]}x{frame_size[1]}")
    return None


def save_calibration(path: Path, calib: CameraCalibration) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format_version": FORMAT_VERSION,
        "created_utc": calib.created_utc or datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "image_width": int(calib.image_width),
        "image_height": int(calib.image_height),
        "camera_matrix": np.asarray(calib.camera_matrix, dtype=float).tolist(),
        "dist_coeffs": np.asarray(calib.dist_coeffs, dtype=float).reshape(1, -1).tolist(),
        "rms_error_px": float(calib.rms_error_px),
        "mean_view_error_px": None if calib.mean_view_error_px is None else float(calib.mean_view_error_px),
        "num_views": int(calib.num_views),
        "pattern": calib.pattern,
    }
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)


def load_calibration(path: Path) -> CameraCalibration:
    """Load a calibration file. Raises CalibrationError if missing or malformed."""
    path = Path(path)
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        return CameraCalibration(
            camera_matrix=np.array(data["camera_matrix"], dtype=np.float64),
            dist_coeffs=np.array(data["dist_coeffs"], dtype=np.float64).reshape(1, -1),
            image_width=int(data["image_width"]),
            image_height=int(data["image_height"]),
            rms_error_px=float(data["rms_error_px"]),
            mean_view_error_px=data.get("mean_view_error_px"),
            num_views=int(data.get("num_views", 0)),
            created_utc=str(data.get("created_utc", "")),
            pattern=dict(data.get("pattern", {})),
        )
    except FileNotFoundError as exc:
        raise CalibrationError(f"Calibration file not found: {path}") from exc
    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        raise CalibrationError(f"Calibration file is invalid ({path}): {exc}") from exc


def try_load_calibration(path: Path) -> Tuple[Optional[CameraCalibration], str]:
    """Load calibration, returning (calibration_or_None, note explaining a failure)."""
    try:
        return load_calibration(path), ""
    except CalibrationError as exc:
        return None, str(exc)


def chessboard_object_points(cols: int, rows: int, square_size_m: float) -> np.ndarray:
    grid = np.zeros((rows * cols, 3), dtype=np.float32)
    grid[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    return grid * square_size_m


def calibrate_from_points(object_points: Sequence[np.ndarray], image_points: Sequence[np.ndarray],
                          image_size: Tuple[int, int],
                          pattern: Optional[Dict[str, Any]] = None) -> CameraCalibration:
    """Run cv2.calibrateCamera and package the result. image_size is (width, height)."""
    import cv2

    if len(object_points) < 3 or len(object_points) != len(image_points):
        raise CalibrationError("Need at least 3 views with matching object/image points")
    obj = [np.asarray(o, dtype=np.float32) for o in object_points]
    img = [np.asarray(i, dtype=np.float32).reshape(-1, 1, 2) for i in image_points]
    try:
        rms, k, d, rvecs, tvecs = cv2.calibrateCamera(obj, img, tuple(image_size), None, None)
    except cv2.error as exc:
        raise CalibrationError(f"cv2.calibrateCamera failed: {exc.msg.strip()[:100]}") from exc

    view_errors: List[float] = []
    for o, i, rv, tv in zip(obj, img, rvecs, tvecs):
        proj, _ = cv2.projectPoints(o, rv, tv, k, d)
        view_errors.append(float(np.sqrt(np.mean(np.sum((proj - i) ** 2, axis=2)))))

    return CameraCalibration(
        camera_matrix=k, dist_coeffs=np.asarray(d, dtype=np.float64).reshape(1, -1),
        image_width=int(image_size[0]), image_height=int(image_size[1]),
        rms_error_px=float(rms), mean_view_error_px=float(np.mean(view_errors)),
        num_views=len(obj),
        created_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        pattern=dict(pattern or {}),
    )
