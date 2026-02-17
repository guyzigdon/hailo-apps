"""Tests for mock/simulation behavior using existing drone_follow APIs.

Uses the same step logic as mock_control_loop: update shared_state with Detection,
compute_velocity_command, apply_physics_step, add pattern drift, clamp. No thread;
we run N steps synchronously.
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


def run_simulation_steps(
    pattern: str,
    initial_x: float,
    initial_y: float,
    target_bbox_height: float,
    num_steps: int,
    period: float = 0.1,
):
    """
    Run N steps of the same logic as mock_control_loop (no drone).
    Returns (shared_state, list of (cx, cy, bh) per step).
    """
    config = ControllerConfig(
        target_bbox_height=target_bbox_height,
        control_loop_hz=1.0 / period,
    )
    state = SharedDetectionState()
    cx, cy = initial_x, initial_y
    bh = target_bbox_height * 0.1
    positions = []
    t0 = time.monotonic()

    for step in range(num_steps):
        t = (step * period) if pattern != "static" else 0.0
        state.update(Detection("person", 0.99, cx, cy, bh, time.monotonic()))
        positions.append((cx, cy, bh))

        det, _ = state.get_latest()
        cmd = compute_velocity_command(det, config)
        cx, cy, bh = apply_physics_step(cx, cy, bh, cmd, period, config)

        dx, dy, dbh = 0.0, 0.0, 0.0
        if pattern == "circle":
            dx = 0.02 * math.cos(2 * math.pi * t / 10.0)
            dbh = 0.008 * math.sin(2 * math.pi * t / 10.0)
        elif pattern == "line":
            dbh = -0.005
        elif pattern == "sweep":
            dx = 0.03 * math.sin(2 * math.pi * t / 5.0)

        cx += dx
        cy += dy
        bh += dbh
        cx = max(0.01, min(0.99, cx))
        cy = max(0.01, min(0.99, cy))
        bh = max(0.02, min(0.95, bh))

    return state, positions


class TestMockCircle:
    def test_produces_detections(self):
        state, _ = run_simulation_steps("circle", 0.5, 0.5, 0.3, num_steps=20)
        det, count = state.get_latest()
        assert count == 20
        assert det is not None
        assert det.label == "person"

    def test_stays_in_bounds(self):
        _, positions = run_simulation_steps("circle", 0.5, 0.5, 0.3, num_steps=50)
        assert len(positions) == 50
        for cx, cy, bh in positions:
            assert 0.0 <= cx <= 1.0
            assert 0.0 <= cy <= 1.0
            assert 0.02 <= bh <= 0.95

    def test_moves_around_center(self):
        """Circle has horizontal and bbox drift; y only changes from drone reaction."""
        _, positions = run_simulation_steps("circle", 0.5, 0.5, 0.3, num_steps=80)
        xs = [p[0] for p in positions]
        bhs = [p[2] for p in positions]
        assert max(xs) - min(xs) > 0.05
        assert max(bhs) - min(bhs) > 0.02


class TestMockSweep:
    def test_sweeps_horizontally(self):
        _, positions = run_simulation_steps("sweep", 0.5, 0.5, 0.3, num_steps=120)
        xs = [p[0] for p in positions]
        assert max(xs) - min(xs) > 0.25

    def test_stays_in_bounds(self):
        _, positions = run_simulation_steps("sweep", 0.5, 0.5, 0.3, num_steps=50)
        for cx, cy, bh in positions:
            assert 0.0 <= cx <= 1.0
            assert 0.0 <= cy <= 1.0
            assert 0.02 <= bh <= 0.95


class TestMockStatic:
    def test_returns_fixed_position(self):
        """Static has no world drift; position still changes from drone reaction (centering)."""
        state, positions = run_simulation_steps("static", 0.7, 0.3, 0.3, num_steps=10)
        assert len(positions) == 10
        det, _ = state.get_latest()
        assert det is not None
        assert det.label == "person"
        assert det.confidence == pytest.approx(0.99)

    def test_static_bbox_initial_smaller_than_target(self):
        """Initial bbox in simulation is target_bbox_height * 0.1 (drone should approach)."""
        _, positions = run_simulation_steps("static", 0.5, 0.5, 0.3, num_steps=1)
        _, _, bh = positions[0]
        assert bh == pytest.approx(0.03, abs=0.01)


class TestMockCommon:
    def test_detection_has_recent_timestamp(self):
        state, _ = run_simulation_steps("circle", 0.5, 0.5, 0.3, num_steps=5)
        det, _ = state.get_latest()
        age = time.monotonic() - det.timestamp
        assert age < 2.0

    @pytest.mark.parametrize("pattern", ["circle", "sweep", "static"])
    def test_all_patterns_produce_confident_detections(self, pattern):
        state, _ = run_simulation_steps(pattern, 0.6, 0.4, 0.3, num_steps=10)
        det, count = state.get_latest()
        assert count == 10
        assert det is not None
        assert det.confidence == pytest.approx(0.99)
        assert det.label == "person"
