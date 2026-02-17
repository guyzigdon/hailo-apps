"""Integration tests: simulation steps -> controller -> velocity commands.

Uses existing APIs only: Detection, SharedDetectionState, ControllerConfig,
compute_velocity_command, apply_physics_step. Same step logic as mock_control_loop.
"""

import math
import time

import pytest

from drone_control import (
    Detection,
    SharedDetectionState,
    ControllerConfig,
    compute_velocity_command,
    apply_physics_step,
)


def run_simulation_and_collect_commands(
    pattern: str,
    initial_x: float,
    initial_y: float,
    target_bbox_height: float,
    num_steps: int,
    period: float = 0.1,
    config: ControllerConfig = None,
):
    """Run N steps (same logic as mock_control_loop), collect velocity command per step."""
    if config is None:
        config = ControllerConfig(target_bbox_height=target_bbox_height)
    state = SharedDetectionState()
    cx, cy = initial_x, initial_y
    bh = target_bbox_height * 0.1
    commands = []

    for step in range(num_steps):
        t = step * period
        state.update(Detection("person", 0.99, cx, cy, bh, time.monotonic()))
        det, _ = state.get_latest()
        if det is not None:
            age = time.monotonic() - det.timestamp
            if age > config.detection_timeout_s:
                det = None
        cmd = compute_velocity_command(det, config)
        commands.append(cmd)
        cx, cy, bh = apply_physics_step(cx, cy, bh, cmd, period, config)

        dx, dy, dbh = 0.0, 0.0, 0.0
        if pattern == "circle":
            dx = 0.02 * math.cos(2 * math.pi * t / 10.0)
            dbh = 0.008 * math.sin(2 * math.pi * t / 10.0)
        elif pattern == "line":
            dbh = -0.005
        elif pattern == "sweep":
            dx = 0.03 * math.sin(2 * math.pi * t / 5.0)
        cx = max(0.01, min(0.99, cx + dx))
        cy = max(0.01, min(0.99, cy + dy))
        bh = max(0.02, min(0.95, bh + dbh))

    return commands


class TestMockToController:
    """Simulation steps -> controller -> velocity commands (existing APIs only)."""

    def test_circle_produces_varying_yaw(self):
        commands = run_simulation_and_collect_commands(
            "circle", 0.7, 0.3, 0.3, num_steps=60, period=0.1
        )
        yaws = [c.yawspeed_deg_s for c in commands]
        yaws_nonzero = [y for y in yaws if abs(y) > 0.1]
        assert len(yaws_nonzero) > 5
        assert any(y > 0 for y in yaws_nonzero)
        assert any(y < 0 for y in yaws_nonzero)

    def test_circle_produces_varying_altitude(self):
        """With fixed_altitude=False and off-center vertical position, we get altitude commands."""
        config = ControllerConfig(fixed_altitude=False)
        commands = run_simulation_and_collect_commands(
            "circle", 0.7, 0.75, 0.3, num_steps=60, period=0.1, config=config
        )
        downs = [c.down_m_s for c in commands]
        downs_nonzero = [d for d in downs if abs(d) > 0.001]
        assert len(downs_nonzero) > 5
        assert any(d > 0 for d in downs_nonzero) or any(d < 0 for d in downs_nonzero)

    def test_circle_produces_varying_forward(self):
        commands = run_simulation_and_collect_commands(
            "circle", 0.7, 0.3, 0.3, num_steps=80, period=0.1
        )
        fwds = [c.forward_m_s for c in commands]
        fwds_nonzero = [f for f in fwds if abs(f) > 0.01]
        assert len(fwds_nonzero) > 3
        assert any(f > 0 for f in fwds_nonzero)
        assert any(f < 0 for f in fwds_nonzero)

    def test_static_starts_with_expected_commands(self):
        """Static off-center (0.7, 0.3) -> initial commands: positive yaw, fly up, forward."""
        config = ControllerConfig(fixed_altitude=False)
        commands = run_simulation_and_collect_commands(
            "static", 0.7, 0.3, 0.3, num_steps=15, period=0.1, config=config
        )
        head = commands[:5]
        assert len(head) == 5
        for cmd in head:
            assert cmd.yawspeed_deg_s > 0.0
            assert cmd.down_m_s < 0.0
            assert cmd.forward_m_s > 0.0

    def test_all_commands_within_limits(self):
        config = ControllerConfig()
        for pattern in ["circle", "sweep", "static"]:
            commands = run_simulation_and_collect_commands(
                pattern, 0.7, 0.3, 0.3, num_steps=30, config=config
            )
            for cmd in commands:
                assert abs(cmd.yawspeed_deg_s) <= config.max_yawspeed + 0.01
                assert abs(cmd.down_m_s) <= config.max_down_speed + 0.01
                assert cmd.forward_m_s <= config.max_forward + 0.01
                assert cmd.forward_m_s >= -config.max_backward - 0.01
                assert cmd.right_m_s == 0.0

    def test_sweep_has_at_least_as_large_yaw_range_as_circle(self):
        """Sweep (±0.35) typically produces similar or larger yaw range than circle (±0.2)."""
        config = ControllerConfig()
        circle_cmds = run_simulation_and_collect_commands(
            "circle", 0.7, 0.3, 0.3, num_steps=60, config=config
        )
        sweep_cmds = run_simulation_and_collect_commands(
            "sweep", 0.7, 0.3, 0.3, num_steps=60, config=config
        )
        circle_max_yaw = max(abs(c.yawspeed_deg_s) for c in circle_cmds)
        sweep_max_yaw = max(abs(c.yawspeed_deg_s) for c in sweep_cmds)
        assert sweep_max_yaw >= circle_max_yaw - 1.0  # sweep at least comparable


class TestStalenessHandling:
    """Stale detections -> search mode (controller API only)."""

    def test_old_detection_triggers_search(self):
        config = ControllerConfig(detection_timeout_s=0.5)
        old_det = Detection(
            label="test", confidence=0.9,
            center_x=0.8, center_y=0.3, bbox_height=0.2,
            timestamp=time.monotonic() - 2.0,
        )
        age = time.monotonic() - old_det.timestamp
        if age > config.detection_timeout_s:
            old_det = None
        cmd = compute_velocity_command(old_det, config)
        assert cmd.yawspeed_deg_s == config.search_yawspeed
        assert cmd.forward_m_s == 0.0

    def test_fresh_detection_used(self):
        config = ControllerConfig(detection_timeout_s=0.5)
        fresh_det = Detection(
            label="test", confidence=0.9,
            center_x=0.8, center_y=0.3, bbox_height=0.2,
            timestamp=time.monotonic(),
        )
        age = time.monotonic() - fresh_det.timestamp
        if age > config.detection_timeout_s:
            fresh_det = None
        cmd = compute_velocity_command(fresh_det, config)
        assert cmd.yawspeed_deg_s != config.search_yawspeed
