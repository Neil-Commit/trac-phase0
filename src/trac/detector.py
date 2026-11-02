"""AprilTag detection (DICT_APRILTAG_36h11) using OpenCV's ArUco module."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import cv2
import numpy as np


@dataclass(frozen=True, eq=False)
class TagDetection:
    tag_id: int
    corners: np.ndarray  # (4, 2) float32 pixels, order: top-left, top-right, bottom-right, bottom-left

    @property
    def center(self) -> np.ndarray:
        return self.corners.mean(axis=0)


class AprilTagDetector:
    def __init__(self) -> None:
        if not hasattr(cv2.aruco, "ArucoDetector"):
            raise RuntimeError(
                f"OpenCV {cv2.__version__} is too old: ArucoDetector needs OpenCV >= 4.7"
            )
        dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
        params = cv2.aruco.DetectorParameters()
        params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_APRILTAG
        self._detector = cv2.aruco.ArucoDetector(dictionary, params)

    def detect(self, frame: np.ndarray) -> List[TagDetection]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        corners, ids, _rejected = self._detector.detectMarkers(gray)
        if ids is None or len(ids) == 0:
            return []
        return [
            TagDetection(int(tag_id), np.asarray(c, dtype=np.float32).reshape(4, 2))
            for tag_id, c in zip(ids.flatten(), corners)
        ]
