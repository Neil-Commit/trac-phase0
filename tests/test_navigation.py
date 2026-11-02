"""Navigation policy tests: synthetic inputs only (no camera, no OpenCV)."""

import math
import random

import pytest

from trac.config import NavigationConfig
from trac.navigation import (STOP_STATES, NavDecision, NavigationPolicy, NavInput, NavState)

CFG = NavigationConfig(
    desired_distance_m=1.0, min_safe_distance_m=0.5, angle_deadband_deg=8.0,
    angle_hysteresis_deg=3.0, distance_hysteresis_m=0.10, smoothing_alpha=1.0,
    required_valid_frames=5, max_linear_mps=0.5, max_angular_rps=1.0,
    linear_gain=0.8, angular_gain=1.5, max_forward_angle_deg=60.0,
)


def valid(angle: float, dist: float) -> NavInput:
    return NavInput(True, True, True, angle, dist)


def settle(policy: NavigationPolicy, inp: NavInput, n: int = None) -> NavDecision:
    d = None
    for _ in range(n or CFG.required_valid_frames):
        d = policy.update(inp)
    return d


@pytest.fixture
def policy() -> NavigationPolicy:
    return NavigationPolicy(CFG)


# ----------------------------------------------------------------- stop states
def test_no_tag_stops(policy):
    d = policy.update(NavInput(False, False, True))
    assert d.state == NavState.NO_TARGET_STOP and d.command.is_zero


def test_wrong_tag_only_stops(policy):
    d = policy.update(NavInput(True, False, True))
    assert d.state == NavState.WRONG_TAG_STOP and d.command.is_zero


def test_uncalibrated_stops_even_with_plausible_numbers(policy):
    d = policy.update(NavInput(True, True, False, 0.0, 2.0, note="no file"))
    assert d.state == NavState.UNCALIBRATED_STOP and d.command.is_zero
    assert "no file" in d.reason


@pytest.mark.parametrize("angle,dist", [
    (None, None), (None, 2.0), (0.0, None),
    (float("nan"), 2.0), (0.0, float("nan")), (float("inf"), 2.0), (0.0, float("inf")),
    (0.0, 0.0), (0.0, -1.0), (95.0, 2.0), (-90.0, 2.0),
])
def test_invalid_pose_stops(policy, angle, dist):
    d = policy.update(NavInput(True, True, True, angle, dist))
    assert d.state == NavState.POSE_INVALID_STOP and d.command.is_zero


def test_too_close_stops(policy):
    d = policy.update(valid(0.0, 0.3))
    assert d.state == NavState.TOO_CLOSE_STOP and d.command.is_zero


def test_too_close_stops_immediately_while_following(policy):
    assert settle(policy, valid(0.0, 2.0)).state == NavState.FOLLOW_FORWARD
    d = policy.update(valid(0.0, 0.4))
    assert d.state == NavState.TOO_CLOSE_STOP and d.command.is_zero


def test_smoothing_never_delays_too_close_stop():
    p = NavigationPolicy(NavigationConfig(**{**CFG.__dict__, "smoothing_alpha": 0.1}))
    settle(p, valid(0.0, 3.0))
    d = p.update(valid(0.0, 0.3))  # smoothed value would still be ~2.7 m
    assert d.state == NavState.TOO_CLOSE_STOP and d.command.is_zero


def test_old_pose_not_used_after_target_loss(policy):
    settle(policy, valid(0.0, 3.0))
    # Stale numbers accidentally still attached to a "no target" frame must be ignored.
    d = policy.update(NavInput(False, False, True, angle_deg=0.0, distance_m=3.0))
    assert d.state == NavState.NO_TARGET_STOP and d.command.is_zero


def test_every_stop_state_has_zero_command():
    inputs = [
        NavInput(False, False, True), NavInput(True, False, True), NavInput(True, True, False, 0, 2),
        NavInput(True, True, True, None, None), valid(0.0, 0.1),
    ]
    for inp in inputs:
        d = NavigationPolicy(CFG).update(inp)
        assert d.state in STOP_STATES and d.command.is_zero


# ------------------------------------------------------- acquisition & loss
def test_requires_consecutive_valid_frames_before_moving(policy):
    for i in range(1, CFG.required_valid_frames):
        d = policy.update(valid(0.0, 2.0))
        assert d.state == NavState.HOLD_POSITION and d.command.is_zero
        assert f"{i}/{CFG.required_valid_frames}" in d.reason
    assert policy.update(valid(0.0, 2.0)).state == NavState.FOLLOW_FORWARD


def test_target_loss_stops_immediately_and_restarts_acquisition(policy):
    assert settle(policy, valid(0.0, 2.0)).command.linear_mps > 0
    assert policy.update(NavInput(False, False, True)).command.is_zero
    d = policy.update(valid(0.0, 2.0))
    assert d.state == NavState.HOLD_POSITION and d.command.is_zero  # re-acquiring


def test_invalid_frame_in_between_resets_count(policy):
    for _ in range(CFG.required_valid_frames - 1):
        policy.update(valid(0.0, 2.0))
    policy.update(NavInput(True, True, True, None, None))
    assert policy.update(valid(0.0, 2.0)).state == NavState.HOLD_POSITION


# ------------------------------------------------------------ movement states
def test_centered_far_follows_forward(policy):
    d = settle(policy, valid(0.0, 2.0))
    assert d.state == NavState.FOLLOW_FORWARD
    assert d.command.linear_mps > 0 and d.command.angular_rps == 0


@pytest.mark.parametrize("angle,state,sign", [
    (25.0, NavState.FOLLOW_RIGHT, -1), (-25.0, NavState.FOLLOW_LEFT, +1),
])
def test_far_and_off_center_follows_with_reduced_speed_and_turn(angle, state, sign):
    straight = settle(NavigationPolicy(CFG), valid(0.0, 2.0))
    d = settle(NavigationPolicy(CFG), valid(angle, 2.0))
    assert d.state == state
    assert 0 < d.command.linear_mps < straight.command.linear_mps
    assert d.command.angular_rps * sign > 0  # right target -> clockwise (negative)


def test_far_and_extremely_off_axis_turns_in_place(policy):
    d = settle(policy, valid(70.0, 2.0))
    assert d.state == NavState.ALIGN_RIGHT
    assert d.command.linear_mps == 0 and d.command.angular_rps < 0


@pytest.mark.parametrize("angle,state,sign", [
    (20.0, NavState.ALIGN_RIGHT, -1), (-20.0, NavState.ALIGN_LEFT, +1),
])
def test_near_off_center_aligns_without_forward_motion(angle, state, sign):
    d = settle(NavigationPolicy(CFG), valid(angle, 0.9))
    assert d.state == state
    assert d.command.linear_mps == 0 and d.command.angular_rps * sign > 0


def test_near_and_centered_holds(policy):
    d = settle(policy, valid(2.0, 0.9))
    assert d.state == NavState.HOLD_POSITION and d.command.is_zero


# ------------------------------------------------------------ limits & safety
def test_commands_are_clamped():
    p = NavigationPolicy(CFG)
    d = settle(p, valid(0.0, 50.0))
    assert d.command.linear_mps == CFG.max_linear_mps
    d = settle(NavigationPolicy(CFG), valid(-59.0, 0.9))
    assert abs(d.command.angular_rps) <= CFG.max_angular_rps


def test_never_reverses_or_exceeds_limits_over_random_inputs():
    rng = random.Random(1234)
    p = NavigationPolicy(NavigationConfig(**{**CFG.__dict__, "smoothing_alpha": 0.5}))
    for _ in range(5000):
        r = rng.random()
        if r < 0.05:
            inp = NavInput(rng.random() < 0.5, rng.random() < 0.5, rng.random() < 0.8,
                           rng.uniform(-120, 120), rng.uniform(-1, 8))
        else:
            inp = valid(rng.uniform(-89, 89), rng.uniform(0.0, 8.0))
        d = p.update(inp)
        c = d.command
        assert 0.0 <= c.linear_mps <= CFG.max_linear_mps
        assert abs(c.angular_rps) <= CFG.max_angular_rps
        assert math.isfinite(c.linear_mps) and math.isfinite(c.angular_rps)
        if d.is_stop:
            assert c.is_zero


# ------------------------------------------------- hysteresis & smoothing
def _transitions(policy, angles, dist=2.0):
    states = [policy.update(valid(a, dist)).state for a in angles]
    return sum(1 for a, b in zip(states, states[1:]) if a != b)


def test_angle_hysteresis_prevents_flicker_when_centered():
    p = NavigationPolicy(CFG)
    settle(p, valid(7.0, 2.0))
    # 7 <-> 10 degrees straddles the 8-degree deadband but stays inside deadband+hysteresis (11)
    assert _transitions(p, [10.0, 7.0] * 20) == 0


def test_angle_hysteresis_prevents_flicker_when_off_center():
    p = NavigationPolicy(CFG)
    settle(p, valid(15.0, 2.0))
    assert _transitions(p, [9.0, 12.0] * 20) == 0  # never <= 8, stays off-center
    assert p.update(valid(7.5, 2.0)).state == NavState.FOLLOW_FORWARD  # crosses back


def test_distance_hysteresis_prevents_flicker_near_desired_distance():
    p = NavigationPolicy(CFG)
    assert settle(p, valid(0.0, 0.95)).state == NavState.HOLD_POSITION
    assert p.update(valid(0.0, 1.05)).state == NavState.HOLD_POSITION  # within desired + 0.10
    assert p.update(valid(0.0, 1.15)).state == NavState.FOLLOW_FORWARD
    assert p.update(valid(0.0, 1.05)).state == NavState.FOLLOW_FORWARD  # keeps following...
    assert p.update(valid(0.0, 0.99)).state == NavState.HOLD_POSITION   # ...until within desired


def test_smoothing_filters_a_single_noisy_frame():
    p = NavigationPolicy(NavigationConfig(**{**CFG.__dict__, "smoothing_alpha": 0.3}))
    settle(p, valid(0.0, 2.0))
    d = p.update(valid(30.0, 2.0))  # one outlier
    assert d.angle_deg == pytest.approx(9.0)
    assert d.state == NavState.FOLLOW_FORWARD  # 9 deg is inside deadband + hysteresis


def test_filter_state_cleared_by_stop():
    p = NavigationPolicy(NavigationConfig(**{**CFG.__dict__, "smoothing_alpha": 0.3}))
    settle(p, valid(40.0, 3.0))
    p.update(NavInput(False, False, True))
    d = p.update(valid(0.0, 1.5))
    assert d.angle_deg == 0.0 and d.distance_m == 1.5  # no memory of the 40 deg / 3 m pose
