"""Navigation decision policy (pure logic).

This module knows nothing about cameras, OpenCV, tag detection, pose estimation, or
display. It maps one frame's summary (``NavInput``) to a simulated decision.

Conventions
-----------
* ``angle_deg``: horizontal angle of the target, ``atan2(x, z)`` in degrees.
  Positive = target is to the camera's RIGHT.
* ``distance_m``: planar distance ``sqrt(x^2 + z^2)`` in metres.
* ``Command.angular_rps``: positive = counter-clockwise (turn LEFT), the usual
  right-hand convention. A target on the right therefore yields a NEGATIVE value.
* ``Command.linear_mps`` is never negative: the simulator never reverses.

Safety behaviour
----------------
* Every stop state returns a zero command and clears all filter state, so an old
  pose can never influence a later decision.
* Movement states are shown only after ``required_valid_frames`` consecutive valid
  frames. Until then the decision is HOLD_POSITION (zero command) with an
  "Acquiring target" reason.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Optional

from .config import NavigationConfig


class NavState(str, Enum):
    NO_TARGET_STOP = "NO_TARGET_STOP"
    WRONG_TAG_STOP = "WRONG_TAG_STOP"
    UNCALIBRATED_STOP = "UNCALIBRATED_STOP"
    POSE_INVALID_STOP = "POSE_INVALID_STOP"
    TOO_CLOSE_STOP = "TOO_CLOSE_STOP"
    ALIGN_LEFT = "ALIGN_LEFT"
    ALIGN_RIGHT = "ALIGN_RIGHT"
    FOLLOW_FORWARD = "FOLLOW_FORWARD"
    FOLLOW_LEFT = "FOLLOW_LEFT"
    FOLLOW_RIGHT = "FOLLOW_RIGHT"
    HOLD_POSITION = "HOLD_POSITION"


STOP_STATES = frozenset({
    NavState.NO_TARGET_STOP,
    NavState.WRONG_TAG_STOP,
    NavState.UNCALIBRATED_STOP,
    NavState.POSE_INVALID_STOP,
    NavState.TOO_CLOSE_STOP,
})


@dataclass(frozen=True)
class Command:
    """Simulated velocity command (never sent to hardware)."""

    linear_mps: float = 0.0
    angular_rps: float = 0.0  # positive = counter-clockwise (left)

    @property
    def is_zero(self) -> bool:
        return self.linear_mps == 0.0 and self.angular_rps == 0.0


ZERO_COMMAND = Command()


@dataclass(frozen=True)
class NavInput:
    """Summary of ONE frame. Values from earlier frames must never be carried in.

    angle_deg / distance_m must be None unless a valid pose was produced from the
    current frame for the assigned tag.
    """

    any_tag_visible: bool
    assigned_tag_visible: bool
    calibrated: bool
    angle_deg: Optional[float] = None
    distance_m: Optional[float] = None
    note: str = ""  # optional detail used in the stop reason (e.g. why uncalibrated)


@dataclass(frozen=True)
class NavDecision:
    state: NavState
    reason: str
    command: Command
    angle_deg: Optional[float] = None     # smoothed values the decision was based on
    distance_m: Optional[float] = None
    valid_frames: int = 0

    @property
    def is_stop(self) -> bool:
        return self.state in STOP_STATES


def _finite(value: Optional[float]) -> bool:
    return value is not None and math.isfinite(value)


def _side(angle_deg: float) -> str:
    return "right" if angle_deg > 0 else "left"


class NavigationPolicy:
    """Stateful per-frame decision maker (smoothing, hysteresis, frame counting)."""

    def __init__(self, cfg: NavigationConfig):
        self._cfg = cfg
        self.reset()

    # ------------------------------------------------------------------ state
    def reset(self) -> None:
        self._valid_frames = 0
        self._angle: Optional[float] = None
        self._dist: Optional[float] = None
        self._centered: Optional[bool] = None
        self._far: Optional[bool] = None

    def _stop(self, state: NavState, reason: str) -> NavDecision:
        self.reset()
        return NavDecision(state, reason, ZERO_COMMAND)

    # ----------------------------------------------------------------- update
    def update(self, inp: NavInput) -> NavDecision:
        cfg = self._cfg

        if not inp.any_tag_visible:
            return self._stop(NavState.NO_TARGET_STOP, inp.note or "No AprilTag detected")
        if not inp.assigned_tag_visible:
            return self._stop(
                NavState.WRONG_TAG_STOP,
                "Only non-assigned tags visible; they are ignored",
            )
        if not inp.calibrated:
            return self._stop(
                NavState.UNCALIBRATED_STOP,
                inp.note or "Camera calibration missing or invalid",
            )

        raw_angle, raw_dist = inp.angle_deg, inp.distance_m
        if (not _finite(raw_angle) or not _finite(raw_dist)
                or raw_dist <= 0 or abs(raw_angle) >= 90):
            return self._stop(NavState.POSE_INVALID_STOP, inp.note or "No valid pose for assigned tag")

        too_close_reason = (
            f"Target {{:.2f}} m is inside the minimum safe distance ({cfg.min_safe_distance_m:.2f} m)"
        )
        # Safety check on the RAW measurement first: smoothing must never delay a stop.
        if raw_dist < cfg.min_safe_distance_m:
            return self._stop(NavState.TOO_CLOSE_STOP, too_close_reason.format(raw_dist))

        self._smooth(raw_angle, raw_dist)
        if self._dist < cfg.min_safe_distance_m:
            return self._stop(NavState.TOO_CLOSE_STOP, too_close_reason.format(self._dist))

        angle, dist = self._angle, self._dist
        self._valid_frames = min(self._valid_frames + 1, cfg.required_valid_frames)
        self._update_flags(angle, dist)

        if self._valid_frames < cfg.required_valid_frames:
            return NavDecision(
                NavState.HOLD_POSITION,
                f"Acquiring target ({self._valid_frames}/{cfg.required_valid_frames} valid frames)",
                ZERO_COMMAND, angle, dist, self._valid_frames,
            )
        return self._decide(angle, dist)

    # --------------------------------------------------------------- internals
    def _smooth(self, angle: float, dist: float) -> None:
        if self._angle is None or self._dist is None:
            self._angle, self._dist = angle, dist
        else:
            a = self._cfg.smoothing_alpha
            self._angle += a * (angle - self._angle)
            self._dist += a * (dist - self._dist)

    def _update_flags(self, angle: float, dist: float) -> None:
        cfg = self._cfg
        off = abs(angle)
        if self._centered is None or not self._centered:
            self._centered = off <= cfg.angle_deadband_deg
        else:  # already centered: leave only beyond deadband + hysteresis
            self._centered = off <= cfg.angle_deadband_deg + cfg.angle_hysteresis_deg

        if self._far is None or self._far:
            self._far = dist > cfg.desired_distance_m
        else:  # already within desired distance: resume only beyond desired + hysteresis
            self._far = dist > cfg.desired_distance_m + cfg.distance_hysteresis_m

    def _clamp(self, linear: float, angular: float) -> Command:
        cfg = self._cfg
        if not (math.isfinite(linear) and math.isfinite(angular)):
            return ZERO_COMMAND
        linear = min(max(linear, 0.0), cfg.max_linear_mps)  # never reverse
        angular = max(-cfg.max_angular_rps, min(cfg.max_angular_rps, angular))
        return Command(linear, angular)

    def _decide(self, angle: float, dist: float) -> NavDecision:
        cfg = self._cfg
        n = self._valid_frames
        side = _side(angle)
        off = abs(angle)
        turn = -cfg.angular_gain * math.radians(angle)  # target right -> negative (clockwise)

        if self._far:
            forward = cfg.linear_gain * (dist - cfg.desired_distance_m)
            if self._centered:
                return NavDecision(
                    NavState.FOLLOW_FORWARD,
                    f"Target centered, {dist:.2f} m away (desired {cfg.desired_distance_m:.2f} m): advancing",
                    self._clamp(forward, 0.0), angle, dist, n,
                )
            if off >= cfg.max_forward_angle_deg:
                return NavDecision(
                    NavState.ALIGN_RIGHT if side == "right" else NavState.ALIGN_LEFT,
                    f"Target {off:.0f} deg to the {side}, too far off-axis to advance: turning in place",
                    self._clamp(0.0, turn), angle, dist, n,
                )
            scale = 1.0 - off / cfg.max_forward_angle_deg
            return NavDecision(
                NavState.FOLLOW_RIGHT if side == "right" else NavState.FOLLOW_LEFT,
                f"Target {off:.0f} deg to the {side} and beyond desired distance: "
                f"advancing slowly while turning {side}",
                self._clamp(forward * scale, turn), angle, dist, n,
            )

        if self._centered:
            return NavDecision(
                NavState.HOLD_POSITION,
                "Target centered within desired distance: holding position",
                ZERO_COMMAND, angle, dist, n,
            )
        return NavDecision(
            NavState.ALIGN_RIGHT if side == "right" else NavState.ALIGN_LEFT,
            f"Target {off:.0f} deg to the {side} within desired distance: turning in place",
            self._clamp(0.0, turn), angle, dist, n,
        )
