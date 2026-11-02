"""Webcam simulator entry point:  python -m trac.app   (or: trac-sim)

Keys:  q / Esc = quit   r = reset navigation filters   h = toggle HUD
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Callable, Optional, Sequence

import numpy as np

from .calibration import try_load_calibration
from .camera import CameraError, FrameSource, WebcamSource
from .config import DEFAULT_CONFIG_PATH, ConfigError, load_config, with_overrides
from .output import CommandSink, SimulatedCommandSink
from .pipeline import FrameResult, TracPipeline

WINDOW = "TRAC Phase 0 - AprilTag navigation simulator"
MAX_CONSECUTIVE_READ_FAILURES = 60


def run_loop(source: FrameSource, pipeline: TracPipeline, sink: CommandSink,
             show: Optional[Callable[[Optional[np.ndarray], FrameResult], int]] = None,
             max_frames: Optional[int] = None) -> int:
    """Core loop. ``show`` displays a frame and returns a key code (or -1); None = headless.

    Returns a process exit code. Kept free of GUI code so it can be driven by tests.
    """
    failures = 0
    frames = 0
    while max_frames is None or frames < max_frames:
        frame = source.read()
        if frame is None:
            failures += 1
            result = pipeline.process_no_frame()   # always a stop
            if failures >= MAX_CONSECUTIVE_READ_FAILURES:
                sink.send(result.decision)
                print("Camera stopped delivering frames; exiting.", file=sys.stderr)
                return 1
        else:
            failures = 0
            result = pipeline.process(frame)
        sink.send(result.decision)
        frames += 1

        if show is not None:
            key = show(frame, result)
            if key in (ord("q"), 27):
                break
            if key == ord("r"):
                pipeline.reset()
    return 0


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="TRAC Phase 0 AprilTag navigation-decision simulator")
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH, help="path to YAML config")
    p.add_argument("--camera", type=int, help="webcam index (overrides config)")
    p.add_argument("--tag-id", type=int, help="assigned tag ID (overrides config)")
    p.add_argument("--tag-size-mm", type=float, help="printed tag size in mm (overrides config)")
    p.add_argument("--verbose", action="store_true", help="print simulated state changes to the console")
    p.add_argument("--headless", action="store_true", help="no window (use with --max-frames, for smoke tests)")
    p.add_argument("--max-frames", type=int, help="exit after this many frames")
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    try:
        cfg = with_overrides(load_config(args.config), camera_index=args.camera,
                             assigned_id=args.tag_id, tag_size_mm=args.tag_size_mm)
    except ConfigError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2

    calibration, load_note = try_load_calibration(cfg.calibration.path)
    if calibration is None:
        print(f"[warn] {load_note}\n[warn] Running UNCALIBRATED: tags are shown, "
              "but distance and simulated movement are disabled. See README: Calibration.")

    try:
        source = WebcamSource(cfg.camera.index, cfg.camera.width, cfg.camera.height,
                              cfg.camera.disable_autofocus)
    except CameraError as exc:
        print(f"Camera error: {exc}", file=sys.stderr)
        return 1

    import cv2
    from . import display

    pipeline = TracPipeline(cfg, calibration, load_note)
    sink = SimulatedCommandSink(log=args.verbose or args.headless)
    state = {"hud": True, "shown": False}

    def show(frame: Optional[np.ndarray], result: FrameResult) -> int:
        img = display.render(frame, result, cfg, state["hud"]) if frame is not None \
            else display.render_no_frame(cfg, result)
        cv2.imshow(WINDOW, img)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("h"):
            state["hud"] = not state["hud"]
        if state["shown"] and cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
            return ord("q")  # window closed with the mouse
        state["shown"] = True
        return key

    try:
        return run_loop(source, pipeline, sink, None if args.headless else show, args.max_frames)
    except KeyboardInterrupt:
        return 0
    finally:
        source.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    raise SystemExit(main())
