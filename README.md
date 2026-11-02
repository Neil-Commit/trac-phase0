# TRAC Phase 0: AprilTag tracking and navigation-decision simulator

First modular prototype for **TRAC**, a person-following robotic tool carrier. It runs on a
laptop with a webcam, detects AprilTags, estimates where the **assigned user's tag** is, and shows
the navigation decision a robot *would* make.

> **Visual simulation only.** Nothing here talks to motors, an ESP32, a Raspberry Pi, LiDAR, or any
> other hardware. All speeds and distances are prototype settings, not final robot specifications.

## What it does

- Detects tags with OpenCV `DICT_APRILTAG_36h11` and shows every detected ID.
- Only the **assigned tag ID** (configurable) can affect navigation; all other IDs are labelled *ignored*.
- With a valid camera calibration, estimates the assigned tag's position, distance, and angle using
  `solvePnP` with `SOLVEPNP_IPPE_SQUARE`.
- Without valid calibration it still detects and labels tags, but shows `UNCALIBRATED` and disables
  metric distance and simulated movement.
- Shows the decision state, the reason, simulated linear/angular commands, and a top-down
  **NAVIGATION DECISION** panel.

## Project layout

```
config/trac_config.yaml     all tunable settings (kept separate from code)
calibration/                camera_calibration.json is written here (git-ignored)
src/trac/
  config.py                 config schema + validation
  camera.py                 FrameSource interface + WebcamSource
  detector.py               AprilTag 36h11 detection
  calibration.py            calibration storage, validation, computation
  pose.py                   solvePnP pose + pose validation
  navigation.py             decision policy (pure logic: no camera/OpenCV/display)
  pipeline.py               wires detect -> pose -> decision for one frame
  output.py                 CommandSink interface + simulated sink
  display.py                overlays, HUD, top-down panel
  app.py                    webcam application / main loop
  calibrate.py              calibration utility
  generate_tags.py          printable tag generator
tests/                      pytest suite (synthetic inputs; no webcam needed)
```

## Setup

Requires Python 3.9+ and a webcam.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e ".[dev]"            # installs runtime deps + pytest and the trac-* commands
```

Without installing the package you can use `pip install -r requirements-dev.txt` and
`PYTHONPATH=src python -m trac.app`.

Dependencies: `opencv-python` (>= 4.7, for the `ArucoDetector` API), `numpy`, `PyYAML`, `Pillow`.
Use the regular `opencv-python` package, not `-headless`, because the app opens a window. On macOS,
allow camera access for your terminal when prompted.

## Run

```bash
python -m trac.app                 # or: trac-sim
python -m trac.app --camera 1 --tag-id 7 --tag-size-mm 80 --verbose
```

| Option | Meaning |
|---|---|
| `--config PATH` | use another YAML config |
| `--camera N` | webcam index (overrides `camera.index`) |
| `--tag-id N` | assigned tag ID (overrides `tag.assigned_id`) |
| `--tag-size-mm X` | printed tag size (overrides `tag.size_mm`) |
| `--verbose` | print simulated state changes to the console |
| `--headless`, `--max-frames N` | no window / stop after N frames (smoke tests) |

**Controls:** `q` or `Esc` quit (closing the window also works) · `r` reset navigation filters ·
`h` hide/show the HUD.

If the camera cannot be opened the app prints a message and exits with code 1; invalid
configuration exits with code 2.

## Tag printing and measurement

```bash
python -m trac.generate_tags --ids 0 1 2 --size-mm 100 --out tags    # or: trac-tags
```

Writes a PDF and PNG per ID (A4 by default; `--page letter` for US Letter), each with the tag, its
ID, and a 50 mm scale bar.

1. Print at **100% / "Actual size"**. Never "fit to page".
2. Check the scale bar measures 50 mm.
3. **Measure the outer black square** (edge to edge, not including the white margin) with a ruler.
4. Put that measurement in `tag.size_mm` in `config/trac_config.yaml`.
5. Mount the tag **flat** on a rigid backing, matte paper preferred (glare breaks detection).

Pose accuracy scales directly with this number: a 5% error in `size_mm` is a 5% error in distance.
The generator rounds to whole pixels per tag module, so the actual size can differ slightly from
`--size-mm` (it prints the actual value). Always trust your ruler.

## Calibration

Metric distance needs the camera's intrinsics. Calibrate once per camera **and per resolution**.

1. Print a chessboard matching `calibration.chessboard_cols` / `chessboard_rows` (**inner** corners,
   default 9x6) and `calibration.square_size_mm` (default 25 mm). Measure the printed squares and
   update the config. Mount the board flat on something rigid.
2. Keep the camera focus **fixed**. Autofocus changes the focal length (the app asks the driver to
   disable it where supported).
3. Run:
   ```bash
   python -m trac.calibrate           # or: trac-calibrate
   ```
   Press `SPACE` to capture when the board is detected. Capture 15+ views at different distances,
   tilts, and positions, including near the image edges and corners. `d` removes the last capture;
   `c` calibrates and saves; `q` quits.
4. Or calibrate from saved photos (all the same resolution): `python -m trac.calibrate --images DIR`.

The result is saved to `calibration/camera_calibration.json` with the camera matrix, distortion
coefficients, calibration resolution, RMS error, per-view error, and board details. It is **not**
saved if the RMS error exceeds `calibration.max_rms_error_px`.

At startup the app marks the camera **UNCALIBRATED** (tags still shown, no distance, zero movement) if
the file is missing or malformed, the matrix is non-finite or implausible, the error is too high, or the
frame resolution differs from the calibration resolution.

## Coordinate convention and pose validation

Camera frame: **+x** tag is to the camera's right, **+z** tag is in front of the camera.
Horizontal angle = `atan2(x, z)` (positive = right). Planar distance = `sqrt(x² + z²)`.

A pose is **rejected** (decision `POSE_INVALID_STOP`) if any value is non-finite, `z <= 0`, the planar
distance is outside `pose.min_distance_m`..`pose.max_distance_m`, or the corner reprojection error exceeds
`pose.max_reprojection_error_px`. Each frame is evaluated on its own; a previous pose is never reused.
Two tags with the same assigned ID in view are treated as ambiguous and also stop.

## Navigation decisions

Evaluated in this order; the first match wins:

| State | When | Command |
|---|---|---|
| `NO_TARGET_STOP` | no tag detected (or no camera frame) | zero |
| `WRONG_TAG_STOP` | tags visible but none has the assigned ID | zero |
| `UNCALIBRATED_STOP` | assigned tag seen, calibration missing/invalid | zero |
| `POSE_INVALID_STOP` | assigned tag seen, pose rejected | zero |
| `TOO_CLOSE_STOP` | distance < `min_safe_distance_m` (checked on the raw value) | zero |
| `FOLLOW_FORWARD` | centered and beyond desired distance | forward |
| `FOLLOW_LEFT` / `FOLLOW_RIGHT` | off-center and beyond desired distance | reduced forward + turn toward target |
| `ALIGN_LEFT` / `ALIGN_RIGHT` | within desired distance but off-center (or too far off-axis to advance) | turn only |
| `HOLD_POSITION` | centered within desired distance, **or still acquiring** | zero |

- **No reverse:** linear velocity is clamped to `0..max_linear_mps`. Angular velocity is clamped to
  `±max_angular_rps`. Positive angular = counter-clockwise (left), so a target on the right gives a
  negative value.
- **Acquisition:** a movement state appears only after `required_valid_frames` consecutive valid
  frames. Until then the decision is `HOLD_POSITION` with reason "Acquiring target (n/N)". Target
  loss or any invalid frame stops immediately and restarts the count.
- **Anti-flicker:** angle and distance are exponentially smoothed (`smoothing_alpha`). Once
  centered, the target stays centered until `|angle| > deadband + hysteresis`. Once within the desired
  distance, following resumes only beyond `desired + distance_hysteresis_m`.
- Stop states clear all filter state, so stale data cannot leak into later decisions.

All numbers are in `config/trac_config.yaml`, which documents each one.

## Tests

```bash
python -m pytest
```

Covers the navigation policy (all states, stops, acquisition, hysteresis, smoothing, clamping, a
randomized never-reverse check), pose validation and `solvePnP` recovery on projected synthetic
corners, calibration validation/round-trip/computation, config validation, an end-to-end pipeline on
rendered synthetic frames (including ignoring non-assigned tags), tag generation, and the main loop with a
fake camera. No webcam is required.

## Limitations

- Visual simulation only: no velocity ever leaves the laptop, and there is no obstacle sensing,
  terrain handling, or path planning.
- A single flat tag gives a noisy pose at long range or steep viewing angles; reliable range depends on
  tag size, camera resolution, lighting, and focus. A single square tag also has an inherent
  planar-pose ambiguity at near-frontal views, which the reprojection check helps catch.
- The tag must be fully visible and flat; glare, motion blur, and partial occlusion cause dropouts, which
  are handled by stopping.
- Calibration is valid only for the same camera, focus, and resolution. Recalibrate if you change them.
- Laptop webcams differ widely; automatic exposure/focus behaviour is driver dependent.
- Control uses distance and bearing only, with no model of the robot's dynamics.

## Future integration points

- **Raspberry Pi camera:** implement `trac.camera.FrameSource` (`read()` / `release()`) for the new camera
  and pass it to `run_loop` in place of `WebcamSource`. Recalibrate for that camera and resolution.
- **ESP32 over UART:** implement `trac.output.CommandSink.send(decision)` as a UART writer in place of
  `SimulatedCommandSink`. `NavDecision.command` (`linear_mps`, `angular_rps`) is the intended payload,
  and stop states already carry a zero command. The motor-side safety layer should treat a lost link
  as a stop.
- The decision policy (`navigation.py`) has no camera, OpenCV, or display dependencies, so it can be
  reused unchanged on the robot.
