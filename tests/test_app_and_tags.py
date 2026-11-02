"""Main-loop and tag-generation tests (fake camera, no window, no webcam)."""

import numpy as np
import pytest

pytest.importorskip("cv2")
import cv2

from trac.app import MAX_CONSECUTIVE_READ_FAILURES, run_loop
from trac.config import AppConfig
from trac.detector import AprilTagDetector
from trac.generate_tags import render_tag_sheet, write_tag_files
from trac.navigation import NavState
from trac.output import SimulatedCommandSink
from trac.pipeline import TracPipeline
from synthetic import make_calibration, render_scene


class FakeSource:
    def __init__(self, frames):
        self._frames, self.released = list(frames), False

    def read(self):
        return self._frames.pop(0) if self._frames else None

    def release(self):
        self.released = True


def test_loop_honours_max_frames_and_simulates_follow():
    calib = make_calibration()
    frame = render_scene(calib, 0.1, [(0, 0.0, 2.0, 15.0)])
    sink = SimulatedCommandSink()
    code = run_loop(FakeSource([frame] * 20), TracPipeline(AppConfig(), calib), sink, None, max_frames=10)
    assert code == 0 and sink.last.state == NavState.FOLLOW_FORWARD


def test_loop_quits_on_q_key():
    frames = [np.full((720, 1280, 3), 255, np.uint8)] * 10
    seen = []
    code = run_loop(FakeSource(frames), TracPipeline(AppConfig(), make_calibration()),
                    SimulatedCommandSink(), lambda f, r: seen.append(1) or ord("q"))
    assert code == 0 and len(seen) == 1


def test_camera_failure_stops_then_exits_nonzero():
    sink = SimulatedCommandSink()
    code = run_loop(FakeSource([]), TracPipeline(AppConfig(), make_calibration()), sink, None)
    assert code == 1 and sink.last.command.is_zero
    assert MAX_CONSECUTIVE_READ_FAILURES > 1


def test_r_key_resets_acquisition():
    calib = make_calibration()
    frame = render_scene(calib, 0.1, [(0, 0.0, 2.0, 15.0)])
    pipe = TracPipeline(AppConfig(), calib)
    keys = iter([-1] * 6 + [ord("r")] + [ord("q")])
    states = []
    run_loop(FakeSource([frame] * 20), pipe, SimulatedCommandSink(),
             lambda f, r: states.append(r.decision.state) or next(keys))
    assert states[5] == NavState.FOLLOW_FORWARD and states[6] == NavState.FOLLOW_FORWARD


@pytest.mark.parametrize("tag_id", [0, 1, 42, 586])
def test_generated_tags_decode_to_their_id(tag_id):
    sheet, actual_mm = render_tag_sheet(tag_id, 60.0, dpi=200)
    assert abs(actual_mm - 60.0) < 1.0
    detections = AprilTagDetector().detect(sheet)
    assert [d.tag_id for d in detections] == [tag_id]


def test_generated_tag_has_requested_physical_size(tmp_path):
    png, pdf, actual = write_tag_files(3, 100.0, tmp_path, dpi=300)
    assert png.exists() and pdf.exists()
    side_px = AprilTagDetector().detect(cv2.imread(str(png)))[0].corners
    width_px = np.linalg.norm(side_px[1] - side_px[0])
    assert width_px / 300 * 25.4 == pytest.approx(actual, rel=0.02)


def test_invalid_tag_requests_rejected():
    with pytest.raises(ValueError):
        render_tag_sheet(587, 100.0)
    with pytest.raises(ValueError):
        render_tag_sheet(0, 400.0)


def test_main_windowed_path_exits_cleanly_and_releases_camera(monkeypatch):
    """Drive main() with a fake webcam and stubbed GUI calls: q key quits, camera is released."""
    import trac.app as app

    source = FakeSource([np.full((720, 1280, 3), 255, np.uint8)] * 50)
    shown, keys = [], iter([-1, -1, ord("h"), -1, ord("q")])
    monkeypatch.setattr(app, "WebcamSource", lambda *a, **k: source)
    monkeypatch.setattr(cv2, "imshow", lambda name, img: shown.append(img.shape))
    monkeypatch.setattr(cv2, "waitKey", lambda ms: next(keys))
    monkeypatch.setattr(cv2, "getWindowProperty", lambda *a: 1.0)
    destroyed = []
    monkeypatch.setattr(cv2, "destroyAllWindows", lambda: destroyed.append(True))

    assert app.main(["--camera", "0"]) == 0
    assert source.released and destroyed
    assert len(shown) == 5 and all(s == (720, 1280, 3) for s in shown)


def test_main_reports_unopenable_camera(monkeypatch, capsys):
    import trac.app as app
    from trac.camera import CameraError

    def boom(*a, **k):
        raise CameraError("no camera")

    monkeypatch.setattr(app, "WebcamSource", boom)
    assert app.main([]) == 1
    assert "Camera error" in capsys.readouterr().err


def test_main_rejects_invalid_config_override(capsys):
    import trac.app as app

    assert app.main(["--tag-id", "9999"]) == 2
    assert "Configuration error" in capsys.readouterr().err
