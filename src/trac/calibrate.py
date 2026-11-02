"""Camera calibration utility (chessboard).  python -m trac.calibrate   (or: trac-calibrate)

Live mode:    SPACE = capture a view   c = calibrate and save   d = delete last   q/Esc = quit
Offline mode: --images DIR calibrates from saved chessboard photos (all the same resolution).

Print a chessboard whose INNER-corner count and square size match config/trac_config.yaml
(calibration.chessboard_cols/rows/square_size_mm), mount it FLAT on something rigid, and
capture 15+ views with different distances, tilts, and positions (include the image edges).
Keep the camera focus fixed (autofocus changes the focal length).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

from .calibration import (CalibrationError, CameraCalibration, calibrate_from_points,
                          chessboard_object_points, save_calibration)
from .camera import CameraError, WebcamSource
from .config import DEFAULT_CONFIG_PATH, CalibrationConfig, ConfigError, load_config, with_overrides

WINDOW = "TRAC calibration"
_SUBPIX = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)


def find_corners(gray: np.ndarray, cols: int, rows: int, fast: bool = False) -> Optional[np.ndarray]:
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
    if fast:
        flags |= cv2.CALIB_CB_FAST_CHECK
    found, corners = cv2.findChessboardCorners(gray, (cols, rows), flags)
    if not found:
        return None
    return cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), _SUBPIX)


def finish_calibration(image_points: List[np.ndarray], image_size: Tuple[int, int],
                       cc: CalibrationConfig, out_path: Path) -> Optional[CameraCalibration]:
    """Calibrate, report, and save if the error is acceptable. Returns the calibration or None."""
    obj = chessboard_object_points(cc.chessboard_cols, cc.chessboard_rows, cc.square_size_m)
    try:
        calib = calibrate_from_points(
            [obj] * len(image_points), image_points, image_size,
            pattern={"type": "chessboard", "cols": cc.chessboard_cols, "rows": cc.chessboard_rows,
                     "square_size_mm": cc.square_size_mm},
        )
    except CalibrationError as exc:
        print(f"Calibration failed: {exc}")
        return None
    print(f"Calibrated {calib.num_views} views at {image_size[0]}x{image_size[1]}: "
          f"RMS {calib.rms_error_px:.3f} px, mean per-view {calib.mean_view_error_px:.3f} px, "
          f"fx={calib.fx:.1f} fy={calib.fy:.1f} cx={calib.cx:.1f} cy={calib.cy:.1f}")
    if calib.rms_error_px > cc.max_rms_error_px:
        print(f"NOT SAVED: RMS error exceeds calibration.max_rms_error_px ({cc.max_rms_error_px}). "
              "Recapture with a flatter board, sharper images, and more varied poses.")
        return None
    save_calibration(out_path, calib)
    print(f"Saved calibration to {out_path}")
    return calib


def calibrate_from_images(directory: Path, cc: CalibrationConfig, out_path: Path) -> int:
    paths = sorted(p for p in directory.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"})
    points: List[np.ndarray] = []
    size: Optional[Tuple[int, int]] = None
    for path in paths:
        gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            continue
        this_size = (gray.shape[1], gray.shape[0])
        if size is not None and this_size != size:
            print(f"Skipping {path.name}: {this_size} differs from {size}")
            continue
        size = size or this_size
        corners = find_corners(gray, cc.chessboard_cols, cc.chessboard_rows)
        print(f"{path.name}: {'chessboard found' if corners is not None else 'no chessboard'}")
        if corners is not None:
            points.append(corners)
    if size is None or len(points) < cc.min_captures:
        print(f"Need at least {cc.min_captures} usable images, found {len(points)}.")
        return 1
    return 0 if finish_calibration(points, size, cc, out_path) else 1


def calibrate_live(cfg, out_path: Path) -> int:
    cc = cfg.calibration
    try:
        source = WebcamSource(cfg.camera.index, cfg.camera.width, cfg.camera.height,
                              cfg.camera.disable_autofocus)
    except CameraError as exc:
        print(f"Camera error: {exc}", file=sys.stderr)
        return 1
    captures: List[np.ndarray] = []
    size: Optional[Tuple[int, int]] = None
    status = ""
    try:
        while True:
            frame = source.read()
            if frame is None:
                continue
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            size = (gray.shape[1], gray.shape[0])
            corners = find_corners(gray, cc.chessboard_cols, cc.chessboard_rows, fast=True)
            view = frame.copy()
            if corners is not None:
                cv2.drawChessboardCorners(view, (cc.chessboard_cols, cc.chessboard_rows), corners, True)
            msg = (f"{len(captures)}/{cc.min_captures}+ captured | board "
                   f"{'FOUND' if corners is not None else 'not found'} | "
                   "SPACE capture, c calibrate, d undo, q quit")
            cv2.putText(view, msg, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(view, msg, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (80, 220, 80), 1, cv2.LINE_AA)
            if status:
                cv2.putText(view, status, (10, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (60, 200, 255), 2, cv2.LINE_AA)
            cv2.imshow(WINDOW, view)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                return 0
            if key == ord(" "):
                if corners is None:
                    status = "No board detected - not captured"
                else:
                    captures.append(corners)
                    status = f"Captured view {len(captures)}"
            elif key == ord("d") and captures:
                captures.pop()
                status = "Removed last capture"
            elif key == ord("c"):
                if len(captures) < cc.min_captures:
                    status = f"Need at least {cc.min_captures} captures"
                elif finish_calibration(captures, size, cc, out_path):
                    return 0
                else:
                    status = "Calibration not saved - see console"
    except KeyboardInterrupt:
        return 0
    finally:
        source.release()
        cv2.destroyAllWindows()


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description="TRAC camera calibration (chessboard)")
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    p.add_argument("--camera", type=int, help="webcam index (overrides config)")
    p.add_argument("--images", type=Path, help="calibrate from a folder of chessboard photos instead of the webcam")
    p.add_argument("--output", type=Path, help="output JSON (default: calibration.file from config)")
    args = p.parse_args(argv)
    try:
        cfg = with_overrides(load_config(args.config), camera_index=args.camera)
    except ConfigError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2
    out_path = args.output or cfg.calibration.path
    if args.images:
        return calibrate_from_images(args.images, cfg.calibration, out_path)
    return calibrate_live(cfg, out_path)


if __name__ == "__main__":
    raise SystemExit(main())
