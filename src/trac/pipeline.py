"""Per-frame pipeline: detect tags -> estimate pose -> build NavInput -> decide.

Each stage lives in its own module; this file only connects them. Only the assigned
tag ID is ever turned into a pose or a NavInput value. Every frame is processed from
scratch: no pose from an earlier frame is ever reused.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from .calibration import CameraCalibration, calibration_problem
from .config import AppConfig
from .detector import AprilTagDetector, TagDetection
from .navigation import NavDecision, NavigationPolicy, NavInput
from .pose import TagPose, compute_pose


def split_detections(detections: List[TagDetection], assigned_id: int
                     ) -> Tuple[List[TagDetection], List[TagDetection]]:
    """Return (detections with the assigned ID, all other detections)."""
    assigned = [d for d in detections if d.tag_id == assigned_id]
    others = [d for d in detections if d.tag_id != assigned_id]
    return assigned, others


@dataclass(frozen=True, eq=False)
class FrameResult:
    frame_size: Optional[Tuple[int, int]]            # (width, height); None if no frame
    detections: List[TagDetection]
    assigned: Optional[TagDetection]
    other_ids: List[int]
    calibrated: bool
    calibration_note: str                            # why uncalibrated ("" when calibrated)
    calibration: Optional[CameraCalibration]
    pose: Optional[TagPose]
    pose_rejection: Optional[str]
    nav_input: NavInput
    decision: NavDecision


class TracPipeline:
    def __init__(self, cfg: AppConfig, calibration: Optional[CameraCalibration],
                 calibration_load_note: str = "",
                 detector: Optional[AprilTagDetector] = None,
                 policy: Optional[NavigationPolicy] = None):
        self._cfg = cfg
        self._calibration = calibration
        self._load_note = calibration_load_note
        self._detector = detector or AprilTagDetector()
        self.policy = policy or NavigationPolicy(cfg.navigation)

    def reset(self) -> None:
        self.policy.reset()

    def process(self, frame: np.ndarray) -> FrameResult:
        height, width = frame.shape[:2]
        frame_size = (width, height)
        detections = self._detector.detect(frame)
        assigned_list, others = split_detections(detections, self._cfg.tag.assigned_id)

        note = calibration_problem(self._calibration, frame_size, self._cfg.calibration.max_rms_error_px)
        if note is not None and self._calibration is None and self._load_note:
            note = self._load_note
        calibrated = note is None

        pose: Optional[TagPose] = None
        rejection: Optional[str] = None
        assigned: Optional[TagDetection] = None
        if len(assigned_list) > 1:
            rejection = "Multiple tags share the assigned ID; target is ambiguous"
        elif assigned_list:
            assigned = assigned_list[0]
            if calibrated:
                result = compute_pose(
                    assigned.corners, assigned.tag_id, self._cfg.tag.size_m,
                    self._calibration.camera_matrix, self._calibration.dist_coeffs,
                    self._cfg.pose,
                )
                pose, rejection = result.pose, result.rejection_reason

        nav_input = NavInput(
            any_tag_visible=bool(detections),
            assigned_tag_visible=bool(assigned_list),
            calibrated=calibrated,
            angle_deg=pose.angle_deg if pose else None,
            distance_m=pose.distance_m if pose else None,
            note=(note if not calibrated else (rejection or "")) or "",
        )
        decision = self.policy.update(nav_input)
        return FrameResult(
            frame_size=frame_size, detections=detections, assigned=assigned,
            other_ids=sorted(d.tag_id for d in others), calibrated=calibrated,
            calibration_note=note or "", calibration=self._calibration if calibrated else None,
            pose=pose, pose_rejection=rejection, nav_input=nav_input, decision=decision,
        )

    def process_no_frame(self) -> FrameResult:
        """Called when the camera delivers no frame: always results in a stop."""
        nav_input = NavInput(False, False, False, note="Camera frame unavailable")
        return FrameResult(
            frame_size=None, detections=[], assigned=None, other_ids=[], calibrated=False,
            calibration_note="Camera frame unavailable", calibration=None, pose=None,
            pose_rejection=None, nav_input=nav_input, decision=self.policy.update(nav_input),
        )
