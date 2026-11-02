"""Configuration loading and validation.

Settings live in ``config/trac_config.yaml``; this module only defines the schema,
defaults, and validation. Unknown keys are rejected so typos do not silently pass.
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Mapping, Optional

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "trac_config.yaml"
APRILTAG_36H11_ID_COUNT = 587  # valid IDs: 0..586


class ConfigError(ValueError):
    """Raised when the configuration file is missing, malformed, or invalid."""


_TYPE_CHECKS = {
    "int": (int,),
    "float": (int, float),
    "bool": (bool,),
    "str": (str,),
}


def _check_types(obj: Any, section: str) -> None:
    for f in fields(obj):
        value = getattr(obj, f.name)
        allowed = _TYPE_CHECKS[str(f.type)]
        if isinstance(value, bool) and bool not in allowed:
            raise ConfigError(f"{section}.{f.name} must be {f.type}, got bool")
        if not isinstance(value, allowed):
            raise ConfigError(f"{section}.{f.name} must be {f.type}, got {type(value).__name__}")
        if isinstance(value, float) and not math.isfinite(value):
            raise ConfigError(f"{section}.{f.name} must be finite")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ConfigError(message)


@dataclass(frozen=True)
class CameraConfig:
    index: int = 0
    width: int = 1280
    height: int = 720
    disable_autofocus: bool = True

    def validate(self) -> None:
        _check_types(self, "camera")
        _require(self.index >= 0, "camera.index must be >= 0")
        _require(self.width > 0 and self.height > 0, "camera.width/height must be positive")


@dataclass(frozen=True)
class TagConfig:
    assigned_id: int = 0
    size_mm: float = 100.0

    @property
    def size_m(self) -> float:
        return self.size_mm / 1000.0

    def validate(self) -> None:
        _check_types(self, "tag")
        _require(
            0 <= self.assigned_id < APRILTAG_36H11_ID_COUNT,
            f"tag.assigned_id must be in 0..{APRILTAG_36H11_ID_COUNT - 1} for DICT_APRILTAG_36h11",
        )
        _require(self.size_mm > 0, "tag.size_mm must be positive")


@dataclass(frozen=True)
class CalibrationConfig:
    file: str = "calibration/camera_calibration.json"
    max_rms_error_px: float = 1.5
    chessboard_cols: int = 9
    chessboard_rows: int = 6
    square_size_mm: float = 25.0
    min_captures: int = 15

    @property
    def path(self) -> Path:
        p = Path(self.file)
        return p if p.is_absolute() else PROJECT_ROOT / p

    @property
    def square_size_m(self) -> float:
        return self.square_size_mm / 1000.0

    def validate(self) -> None:
        _check_types(self, "calibration")
        _require(self.max_rms_error_px > 0, "calibration.max_rms_error_px must be positive")
        _require(
            self.chessboard_cols >= 3 and self.chessboard_rows >= 3,
            "calibration.chessboard_cols/rows must be >= 3 (inner corners)",
        )
        _require(
            self.chessboard_cols != self.chessboard_rows,
            "calibration.chessboard_cols and rows must differ (square patterns are orientation-ambiguous)",
        )
        _require(self.square_size_mm > 0, "calibration.square_size_mm must be positive")
        _require(self.min_captures >= 5, "calibration.min_captures must be >= 5")


@dataclass(frozen=True)
class PoseConfig:
    min_distance_m: float = 0.10
    max_distance_m: float = 6.0
    max_reprojection_error_px: float = 3.0

    def validate(self) -> None:
        _check_types(self, "pose")
        _require(self.min_distance_m >= 0, "pose.min_distance_m must be >= 0")
        _require(self.max_distance_m > self.min_distance_m, "pose.max_distance_m must exceed min_distance_m")
        _require(self.max_reprojection_error_px > 0, "pose.max_reprojection_error_px must be positive")


@dataclass(frozen=True)
class NavigationConfig:
    desired_distance_m: float = 1.0
    min_safe_distance_m: float = 0.5
    angle_deadband_deg: float = 8.0
    angle_hysteresis_deg: float = 3.0
    distance_hysteresis_m: float = 0.10
    smoothing_alpha: float = 0.4
    required_valid_frames: int = 5
    max_linear_mps: float = 0.5
    max_angular_rps: float = 1.0
    linear_gain: float = 0.8
    angular_gain: float = 1.5
    max_forward_angle_deg: float = 60.0

    def validate(self) -> None:
        _check_types(self, "navigation")
        _require(self.min_safe_distance_m > 0, "navigation.min_safe_distance_m must be positive")
        _require(
            self.desired_distance_m > self.min_safe_distance_m,
            "navigation.desired_distance_m must exceed min_safe_distance_m",
        )
        _require(self.angle_deadband_deg >= 0, "navigation.angle_deadband_deg must be >= 0")
        _require(self.angle_hysteresis_deg >= 0, "navigation.angle_hysteresis_deg must be >= 0")
        _require(self.distance_hysteresis_m >= 0, "navigation.distance_hysteresis_m must be >= 0")
        _require(0 < self.smoothing_alpha <= 1, "navigation.smoothing_alpha must be in (0, 1]")
        _require(self.required_valid_frames >= 1, "navigation.required_valid_frames must be >= 1")
        _require(self.max_linear_mps > 0, "navigation.max_linear_mps must be positive")
        _require(self.max_angular_rps > 0, "navigation.max_angular_rps must be positive")
        _require(self.linear_gain > 0 and self.angular_gain > 0, "navigation gains must be positive")
        _require(
            0 < self.max_forward_angle_deg <= 90,
            "navigation.max_forward_angle_deg must be in (0, 90]",
        )
        _require(
            self.angle_deadband_deg + self.angle_hysteresis_deg < self.max_forward_angle_deg,
            "navigation.angle_deadband_deg + angle_hysteresis_deg must be below max_forward_angle_deg",
        )


@dataclass(frozen=True)
class AppConfig:
    camera: CameraConfig = field(default_factory=CameraConfig)
    tag: TagConfig = field(default_factory=TagConfig)
    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)
    pose: PoseConfig = field(default_factory=PoseConfig)
    navigation: NavigationConfig = field(default_factory=NavigationConfig)

    def validate(self) -> None:
        for section in (self.camera, self.tag, self.calibration, self.pose, self.navigation):
            section.validate()
        _require(
            self.navigation.desired_distance_m <= self.pose.max_distance_m,
            "navigation.desired_distance_m must not exceed pose.max_distance_m",
        )


def _build_section(cls: type, data: Optional[Mapping[str, Any]], name: str):
    if data is None:
        data = {}
    if not isinstance(data, Mapping):
        raise ConfigError(f"Section '{name}' must be a mapping")
    unknown = set(data) - {f.name for f in fields(cls)}
    if unknown:
        raise ConfigError(f"Unknown key(s) in '{name}': {sorted(unknown)}")
    return cls(**data)


def config_from_dict(data: Optional[Mapping[str, Any]]) -> AppConfig:
    """Build and validate an AppConfig from a plain mapping (e.g. parsed YAML)."""
    data = data or {}
    if not isinstance(data, Mapping):
        raise ConfigError("Top level of the config must be a mapping")
    sections = {f.name: f.default_factory for f in fields(AppConfig)}
    unknown = set(data) - set(sections)
    if unknown:
        raise ConfigError(f"Unknown config section(s): {sorted(unknown)}")
    cfg = AppConfig(
        camera=_build_section(CameraConfig, data.get("camera"), "camera"),
        tag=_build_section(TagConfig, data.get("tag"), "tag"),
        calibration=_build_section(CalibrationConfig, data.get("calibration"), "calibration"),
        pose=_build_section(PoseConfig, data.get("pose"), "pose"),
        navigation=_build_section(NavigationConfig, data.get("navigation"), "navigation"),
    )
    cfg.validate()
    return cfg


def load_config(path: Optional[Path] = None) -> AppConfig:
    """Load configuration from YAML (default: config/trac_config.yaml)."""
    path = Path(path) if path else DEFAULT_CONFIG_PATH
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
    except FileNotFoundError as exc:
        raise ConfigError(f"Config file not found: {path}") from exc
    except yaml.YAMLError as exc:
        raise ConfigError(f"Config file is not valid YAML: {exc}") from exc
    return config_from_dict(data)


def with_overrides(cfg: AppConfig, *, camera_index: Optional[int] = None,
                   assigned_id: Optional[int] = None,
                   tag_size_mm: Optional[float] = None) -> AppConfig:
    """Return a validated copy of cfg with command-line overrides applied."""
    camera, tag = cfg.camera, cfg.tag
    if camera_index is not None:
        camera = dataclasses.replace(camera, index=camera_index)
    if assigned_id is not None:
        tag = dataclasses.replace(tag, assigned_id=assigned_id)
    if tag_size_mm is not None:
        tag = dataclasses.replace(tag, size_mm=tag_size_mm)
    new_cfg = dataclasses.replace(cfg, camera=camera, tag=tag)
    new_cfg.validate()
    return new_cfg
