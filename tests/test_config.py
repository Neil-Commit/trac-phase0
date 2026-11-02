import pytest

from trac.config import ConfigError, config_from_dict, load_config, with_overrides


def test_shipped_config_is_valid():
    cfg = load_config()
    assert cfg.tag.assigned_id >= 0 and cfg.navigation.min_safe_distance_m < cfg.navigation.desired_distance_m


def test_empty_config_uses_defaults():
    assert config_from_dict({}).navigation.required_valid_frames == 5


@pytest.mark.parametrize("data", [
    {"navigation": {"typo_key": 1}},
    {"nope": {}},
    {"navigation": {"min_safe_distance_m": 1.5, "desired_distance_m": 1.0}},
    {"navigation": {"smoothing_alpha": 0.0}},
    {"navigation": {"smoothing_alpha": 1.5}},
    {"navigation": {"required_valid_frames": 0}},
    {"navigation": {"max_linear_mps": -1}},
    {"navigation": {"desired_distance_m": float("nan")}},
    {"navigation": {"desired_distance_m": "far"}},
    {"navigation": {"angle_deadband_deg": 58.0, "angle_hysteresis_deg": 5.0}},
    {"tag": {"assigned_id": 587}},
    {"tag": {"assigned_id": -1}},
    {"tag": {"size_mm": 0}},
    {"tag": {"assigned_id": True}},
    {"pose": {"min_distance_m": 5.0, "max_distance_m": 2.0}},
    {"calibration": {"chessboard_cols": 6, "chessboard_rows": 6}},
    {"navigation": {"desired_distance_m": 9.0}},
])
def test_invalid_configs_rejected(data):
    with pytest.raises(ConfigError):
        config_from_dict(data)


def test_missing_file_reports_config_error(tmp_path):
    with pytest.raises(ConfigError):
        load_config(tmp_path / "missing.yaml")


def test_overrides_are_applied_and_validated():
    cfg = with_overrides(load_config(), camera_index=2, assigned_id=7, tag_size_mm=80)
    assert (cfg.camera.index, cfg.tag.assigned_id, cfg.tag.size_m) == (2, 7, 0.08)
    with pytest.raises(ConfigError):
        with_overrides(load_config(), assigned_id=9999)
