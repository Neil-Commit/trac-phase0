"""All drawing code. Reads results; never changes decisions."""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

import cv2
import numpy as np

from .config import AppConfig
from .navigation import STOP_STATES, NavState
from .pipeline import FrameResult

Color = Tuple[int, int, int]
GREEN: Color = (80, 210, 80)
RED: Color = (60, 60, 235)
YELLOW: Color = (60, 220, 240)
ORANGE: Color = (60, 160, 255)
GRAY: Color = (170, 170, 170)
WHITE: Color = (240, 240, 240)
CYAN: Color = (230, 210, 60)
FONT = cv2.FONT_HERSHEY_SIMPLEX


def state_color(state: NavState) -> Color:
    if state in STOP_STATES:
        return RED
    if state == NavState.HOLD_POSITION:
        return YELLOW
    return GREEN


def _text(img: np.ndarray, text: str, org: Tuple[int, int], scale: float, color: Color,
          thickness: int = 1) -> None:
    cv2.putText(img, text, org, FONT, scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
    cv2.putText(img, text, org, FONT, scale, color, thickness, cv2.LINE_AA)


def _blend_rect(img: np.ndarray, p1: Tuple[int, int], p2: Tuple[int, int], color: Color,
                alpha: float) -> None:
    x1, y1 = max(p1[0], 0), max(p1[1], 0)
    x2, y2 = min(p2[0], img.shape[1]), min(p2[1], img.shape[0])
    if x2 <= x1 or y2 <= y1:
        return
    roi = img[y1:y2, x1:x2]
    img[y1:y2, x1:x2] = cv2.addWeighted(roi, 1 - alpha, np.full_like(roi, color), alpha, 0)


# --------------------------------------------------------------------- camera overlay
def draw_tags(img: np.ndarray, result: FrameResult, assigned_id: int) -> None:
    scale = max(0.5, img.shape[1] / 1280 * 0.7)
    for det in result.detections:
        is_assigned = det.tag_id == assigned_id
        color = GREEN if is_assigned else ORANGE
        pts = det.corners.astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(img, [pts], True, color, 3 if is_assigned else 2, cv2.LINE_AA)
        cx, cy = (int(v) for v in det.center)
        cv2.circle(img, (cx, cy), 5, color, -1, cv2.LINE_AA)
        label = f"ID {det.tag_id}" + (" (ASSIGNED)" if is_assigned else " (ignored)")
        top = int(det.corners[:, 1].min())
        _text(img, label, (int(det.corners[:, 0].min()), max(top - 8, 18)), scale * 0.8, color, 2)

    if result.pose is not None and result.calibration is not None:
        try:
            cv2.drawFrameAxes(img, result.calibration.camera_matrix, result.calibration.dist_coeffs,
                              result.pose.rvec, result.pose.tvec, 0.05, 2)
        except cv2.error:
            pass


def draw_centerline(img: np.ndarray, result: FrameResult) -> None:
    h, w = img.shape[:2]
    # The optical axis (principal point) defines "angle = 0"; fall back to image centre.
    x = int(round(result.calibration.cx)) if result.calibration is not None else w // 2
    for y in range(0, h, 20):
        cv2.line(img, (x, y), (x, min(y + 10, h)), CYAN, 1, cv2.LINE_AA)


def draw_hud(img: np.ndarray, result: FrameResult, cfg: AppConfig) -> None:
    h, w = img.shape[:2]
    scale = max(0.5, w / 1280 * 0.62)
    line_h = int(30 * scale / 0.62 * 0.62) + 6
    d = result.decision
    pose = result.pose

    lines: List[Tuple[str, Color]] = []
    if result.calibrated:
        c = result.calibration
        lines.append((f"CALIBRATION: OK ({c.image_width}x{c.image_height}, RMS {c.rms_error_px:.2f} px)", GREEN))
    else:
        lines.append(("UNCALIBRATED: metric distance and simulated movement disabled", RED))
        lines.append((f"  {result.calibration_note}", RED))

    if result.assigned is not None:
        lines.append((f"ASSIGNED TAG {cfg.tag.assigned_id}: FOUND", GREEN))
    else:
        lines.append((f"ASSIGNED TAG {cfg.tag.assigned_id}: NOT FOUND", RED))
    if result.other_ids:
        lines.append((f"Other tags (ignored): {', '.join(map(str, result.other_ids))}", ORANGE))

    if pose is not None:
        lines.append((f"x {pose.x_m:+.2f} m   z {pose.z_m:.2f} m   dist {pose.distance_m:.2f} m", WHITE))
        lines.append((f"angle {pose.angle_deg:+.1f} deg   reproj err {pose.reprojection_error_px:.2f} px", WHITE))
    else:
        why = "disabled (uncalibrated)" if not result.calibrated else (result.pose_rejection or "no pose")
        lines.append((f"x --   z --   dist --   angle --   reproj err --   [{why}]", GRAY))

    lines.append((f"STATE: {d.state.value}", state_color(d.state)))
    lines.append((f"REASON: {d.reason}", WHITE))
    lines.append((f"SIM CMD  linear {d.command.linear_mps:.2f} m/s   angular {d.command.angular_rps:+.2f} rad/s", WHITE))
    lines.append(("SIMULATION ONLY - no hardware output", GRAY))

    pad = 10
    box_w = min(w - 2 * pad, int(max(cv2.getTextSize(t, FONT, scale, 1)[0][0] for t, _ in lines)) + 2 * pad)
    box_h = len(lines) * line_h + pad
    _blend_rect(img, (pad, pad), (pad + box_w, pad + box_h), (20, 20, 20), 0.6)
    for i, (text, color) in enumerate(lines):
        _text(img, text, (pad * 2, pad + (i + 1) * line_h - 4), scale, color, 1)


# ------------------------------------------------------------------ top-down panel
def draw_nav_panel(img: np.ndarray, result: FrameResult, cfg: AppConfig) -> None:
    """Small top-down view. Shows inputs (tag position) and outputs (decision) only."""
    h, w = img.shape[:2]
    size = int(min(h * 0.45, w * 0.4, 340))
    if size < 160:
        return
    margin = 10
    x0, y0 = w - size - margin, h - size - margin
    _blend_rect(img, (x0, y0), (x0 + size, y0 + size), (25, 25, 25), 0.75)
    cv2.rectangle(img, (x0, y0), (x0 + size, y0 + size), GRAY, 1)
    _text(img, "NAVIGATION DECISION", (x0 + 8, y0 + 20), 0.5, WHITE, 1)

    nav = cfg.navigation
    d = result.decision
    pose = result.pose
    color = state_color(d.state)

    robot = (x0 + size // 2, y0 + size - 62)
    view_range_m = max(2.0 * nav.desired_distance_m, 2.0)
    if pose is not None:  # grow the view so the tag stays visible; 0.5 m steps avoid jitter
        view_range_m = max(view_range_m, math.ceil(pose.z_m * 1.1 * 2) / 2)
    px_per_m = (robot[1] - (y0 + 34)) / view_range_m

    # Desired-distance zone and minimum-safe arc
    zone = np.zeros_like(img[y0:y0 + size, x0:x0 + size])
    r_des = int(nav.desired_distance_m * px_per_m)
    cv2.ellipse(zone, (robot[0] - x0, robot[1] - y0), (r_des, r_des), 0, 180, 360, (70, 140, 70), -1)
    roi = img[y0:y0 + size, x0:x0 + size]
    mask = zone.any(axis=2)
    roi[mask] = cv2.addWeighted(roi, 0.6, zone, 0.4, 0)[mask]
    cv2.ellipse(img, robot, (r_des, r_des), 0, 180, 360, GREEN, 1, cv2.LINE_AA)
    r_min = int(nav.min_safe_distance_m * px_per_m)
    cv2.ellipse(img, robot, (r_min, r_min), 0, 180, 360, RED, 1, cv2.LINE_AA)
    _text(img, f"{nav.desired_distance_m:.1f} m", (robot[0] + r_des - 42, robot[1] - 6), 0.4, GREEN, 1)

    # Robot (triangle pointing forward/up)
    tri = np.array([[robot[0], robot[1] - 12], [robot[0] - 9, robot[1] + 9], [robot[0] + 9, robot[1] + 9]], np.int32)
    cv2.fillPoly(img, [tri], WHITE, cv2.LINE_AA)

    # Tag (input): only when a valid current pose exists
    if pose is not None:
        tx, ty = robot[0] + pose.x_m * px_per_m, robot[1] - pose.z_m * px_per_m
        cx = int(min(max(tx, x0 + 8), x0 + size - 8))
        cy = int(min(max(ty, y0 + 30), y0 + size - 8))
        clipped = (cx, cy) != (int(tx), int(ty))
        cv2.rectangle(img, (cx - 7, cy - 7), (cx + 7, cy + 7), GREEN, 2 if clipped else -1)
    else:
        msg = "NO METRIC POSE" if not result.calibrated else "NO VALID TAG POSE"
        _text(img, msg, (x0 + 8, y0 + 42), 0.42, GRAY, 1)

    # Direction (output)
    cmd = d.command
    if d.is_stop:
        cv2.circle(img, robot, 20, RED, 2, cv2.LINE_AA)
        _text(img, "STOP", (robot[0] + 26, robot[1] + 6), 0.55, RED, 2)
    elif cmd.linear_mps > 0:
        length = 18 + 0.3 * size * cmd.linear_mps / nav.max_linear_mps
        theta = 0.6 * cmd.angular_rps / nav.max_angular_rps  # visual steer angle; + = left
        start = (robot[0], robot[1] - 12)
        tip = (int(start[0] - math.sin(theta) * length), int(start[1] - math.cos(theta) * length))
        cv2.arrowedLine(img, start, tip, color, 3, cv2.LINE_AA, tipLength=0.3)
    elif cmd.angular_rps != 0:
        r = 26
        left = cmd.angular_rps > 0

        def pt(deg: float) -> Tuple[int, int]:
            return int(robot[0] + r * math.cos(math.radians(deg))), int(robot[1] + r * math.sin(math.radians(deg)))

        if left:
            cv2.ellipse(img, robot, (r, r), 0, 205, 270, color, 3, cv2.LINE_AA)
            cv2.arrowedLine(img, pt(215), pt(198), color, 3, cv2.LINE_AA, tipLength=0.8)
        else:
            cv2.ellipse(img, robot, (r, r), 0, 270, 335, color, 3, cv2.LINE_AA)
            cv2.arrowedLine(img, pt(325), pt(342), color, 3, cv2.LINE_AA, tipLength=0.8)
    else:
        _text(img, "HOLD", (robot[0] + 18, robot[1] + 6), 0.55, YELLOW, 2)

    # Footer: inputs and outputs only
    if pose is not None:
        _text(img, f"IN  x {pose.x_m:+.2f}  z {pose.z_m:.2f} m", (x0 + 8, y0 + size - 26), 0.42, WHITE, 1)
    else:
        _text(img, "IN  --", (x0 + 8, y0 + size - 26), 0.42, GRAY, 1)
    _text(img, f"OUT {d.state.value}", (x0 + 8, y0 + size - 8), 0.42, color, 1)


# ------------------------------------------------------------------------- entry point
def render(frame: np.ndarray, result: FrameResult, cfg: AppConfig, show_hud: bool = True) -> np.ndarray:
    """Return an annotated copy of the frame."""
    img = frame.copy()
    draw_centerline(img, result)
    draw_tags(img, result, cfg.tag.assigned_id)
    if show_hud:
        draw_hud(img, result, cfg)
        draw_nav_panel(img, result, cfg)
    return img


def render_no_frame(cfg: AppConfig, result: FrameResult) -> np.ndarray:
    img = np.zeros((cfg.camera.height, cfg.camera.width, 3), dtype=np.uint8)
    _text(img, "CAMERA FRAME UNAVAILABLE", (20, 50), 0.9, RED, 2)
    _text(img, f"STATE: {result.decision.state.value} (zero command)", (20, 90), 0.7, RED, 2)
    return img
