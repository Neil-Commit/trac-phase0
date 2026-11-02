"""Frame sources.

``FrameSource`` is the seam for a future Raspberry Pi camera: implement ``read`` and
``release`` for the new camera and pass it to the pipeline/app unchanged.
"""

from __future__ import annotations

from typing import Optional, Protocol

import cv2
import numpy as np


class CameraError(RuntimeError):
    """Raised when the camera cannot be opened."""


class FrameSource(Protocol):
    def read(self) -> Optional[np.ndarray]:
        """Return the next BGR frame, or None if no frame is available."""

    def release(self) -> None:
        """Free the underlying device."""


class WebcamSource:
    """Laptop webcam via cv2.VideoCapture."""

    def __init__(self, index: int, width: int, height: int, disable_autofocus: bool = True):
        self._cap = cv2.VideoCapture(index)
        if not self._cap.isOpened():
            self._cap.release()
            raise CameraError(
                f"Could not open camera index {index}. Try another index (--camera N) "
                "and check that no other application is using the webcam."
            )
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        if disable_autofocus:
            self._cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)  # not supported by every driver; ignored then

    def read(self) -> Optional[np.ndarray]:
        ok, frame = self._cap.read()
        return frame if ok and frame is not None else None

    def release(self) -> None:
        self._cap.release()
