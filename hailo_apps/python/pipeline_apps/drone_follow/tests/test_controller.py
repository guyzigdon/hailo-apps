"""Tests for the FOV-aware proportional controller."""

import time

import pytest
from mavsdk.offboard import VelocityBodyYawspeed

from drone_control import (
    Detection,
    ControllerConfig,
    compute_velocity_command,
    apply_physics_step,
)


def _det(cx=0.5, cy=0.5, bh=0.3):
    """Helper to create a Detection at given normalized coords."""
    return Detection(
        label="test", confidence=0.9,
        center_x=cx, center_y=cy, bbox_height=bh,
        timestamp=time.monotonic(),
    )


@pytest.fixture
def config():
    return ControllerConfig()


# ---- No detection (search mode) ----

class TestSearchMode:
    def test_no_detection_returns_search_yaw(self, config):
        cmd = compute_velocity_command(None, config)
        assert cmd.yawspeed_deg_s == config.search_yawspeed

    def test_no_detection_zero_velocity(self, config):
        cmd = compute_velocity_command(None, config)
        assert cmd.forward_m_s == 0.0
        assert cmd.right_m_s == 0.0
        assert cmd.down_m_s == 0.0


# ---- Yaw (horizontal centering) ----

class TestYaw:
    def test_centered_within_dead_zone(self, config):
        """Detection near center -> zero yaw (dead zone)."""
        cmd = compute_velocity_command(_det(cx=0.51), config)
        assert cmd.yawspeed_deg_s == 0.0

    def test_target_right_positive_yaw(self, config):
        """Detection right of center -> positive yaw (clockwise)."""
        cmd = compute_velocity_command(_det(cx=0.75), config)
        assert cmd.yawspeed_deg_s > 0.0

    def test_target_left_negative_yaw(self, config):
        """Detection left of center -> negative yaw (counter-clockwise)."""
        cmd = compute_velocity_command(_det(cx=0.25), config)
        assert cmd.yawspeed_deg_s < 0.0

    def test_symmetry(self, config):
        """Equal offsets left and right should produce equal magnitude."""
        cmd_right = compute_velocity_command(_det(cx=0.7), config)
        cmd_left = compute_velocity_command(_det(cx=0.3), config)
        assert abs(cmd_right.yawspeed_deg_s + cmd_left.yawspeed_deg_s) < 0.01

    def test_yaw_saturation(self, config):
        """Extreme offset should be clamped to max_yawspeed."""
        cmd = compute_velocity_command(_det(cx=1.0), config)
        assert abs(cmd.yawspeed_deg_s) <= config.max_yawspeed + 0.01

    def test_fov_scaling(self):
        """Wider FOV with same pixel offset -> larger angular error -> larger yaw rate."""
        narrow = ControllerConfig(hfov=60.0)
        wide = ControllerConfig(hfov=120.0)
        det = _det(cx=0.7)
        cmd_narrow = compute_velocity_command(det, narrow)
        cmd_wide = compute_velocity_command(det, wide)
        assert abs(cmd_wide.yawspeed_deg_s) > abs(cmd_narrow.yawspeed_deg_s)

    def test_fov_proportional(self):
        """Double the FOV should double the angular error and thus the yaw rate
        (when not saturated)."""
        cfg_a = ControllerConfig(hfov=40.0, max_yawspeed=9999.0)
        cfg_b = ControllerConfig(hfov=80.0, max_yawspeed=9999.0)
        det = _det(cx=0.6)  # small offset to stay in linear region
        cmd_a = compute_velocity_command(det, cfg_a)
        cmd_b = compute_velocity_command(det, cfg_b)
        ratio = cmd_b.yawspeed_deg_s / cmd_a.yawspeed_deg_s
        assert abs(ratio - 2.0) < 0.01


# ---- Altitude (vertical centering) ----
# Default config has fixed_altitude=True (down always 0). Use fixed_altitude=False for altitude tests.

class TestAltitude:
    def test_centered_within_dead_zone(self, config):
        cmd = compute_velocity_command(_det(cy=0.51), config)
        assert cmd.down_m_s == 0.0

    def test_target_below_positive_down(self):
        """Target below center -> fly down (positive down_m_s)."""
        config = ControllerConfig(fixed_altitude=False)
        cmd = compute_velocity_command(_det(cy=0.75), config)
        assert cmd.down_m_s > 0.0

    def test_target_above_negative_down(self):
        """Target above center -> fly up (negative down_m_s)."""
        config = ControllerConfig(fixed_altitude=False)
        cmd = compute_velocity_command(_det(cy=0.25), config)
        assert cmd.down_m_s < 0.0

    def test_altitude_saturation(self):
        config = ControllerConfig(fixed_altitude=False)
        cmd = compute_velocity_command(_det(cy=1.0), config)
        assert abs(cmd.down_m_s) <= config.max_down_speed + 0.01

    def test_vfov_scaling(self):
        """Wider vertical FOV -> larger altitude command for same pixel offset."""
        narrow = ControllerConfig(vfov=30.0, fixed_altitude=False)
        wide = ControllerConfig(vfov=90.0, fixed_altitude=False)
        det = _det(cy=0.7)
        cmd_narrow = compute_velocity_command(det, narrow)
        cmd_wide = compute_velocity_command(det, wide)
        assert abs(cmd_wide.down_m_s) > abs(cmd_narrow.down_m_s)


# ---- Forward/backward (distance via bbox height) ----

class TestForward:
    def test_at_target_height_in_dead_zone(self, config):
        """Bbox height == target -> no forward movement (dead zone)."""
        cmd = compute_velocity_command(
            _det(bh=config.target_bbox_height), config
        )
        assert cmd.forward_m_s == 0.0

    def test_small_bbox_forward(self, config):
        """Small bbox (far away) -> fly forward."""
        cmd = compute_velocity_command(_det(bh=0.1), config)
        assert cmd.forward_m_s > 0.0

    def test_large_bbox_backward(self, config):
        """Large bbox (too close) -> fly backward."""
        cmd = compute_velocity_command(_det(bh=0.6), config)
        assert cmd.forward_m_s < 0.0

    def test_forward_saturation(self, config):
        """Very small bbox -> clamped to max_forward."""
        cmd = compute_velocity_command(_det(bh=0.01), config)
        assert cmd.forward_m_s <= config.max_forward + 0.01

    def test_backward_saturation(self, config):
        """Very large bbox -> clamped to max_backward."""
        cmd = compute_velocity_command(_det(bh=0.95), config)
        assert cmd.forward_m_s >= -config.max_backward - 0.01

    def test_height_dead_zone(self, config):
        """Bbox slightly off target but within dead zone -> zero forward."""
        dead_zone = (config.dead_zone_height_percent / 100.0) * config.target_bbox_height
        small_offset = dead_zone * 0.5
        cmd = compute_velocity_command(
            _det(bh=config.target_bbox_height + small_offset), config
        )
        assert cmd.forward_m_s == 0.0

    def test_right_always_zero(self, config):
        """right_m_s should always be zero (no lateral movement)."""
        for cx in [0.1, 0.5, 0.9]:
            for cy in [0.1, 0.5, 0.9]:
                for bh in [0.1, 0.3, 0.8]:
                    cmd = compute_velocity_command(_det(cx=cx, cy=cy, bh=bh), config)
                    assert cmd.right_m_s == 0.0


# ---- Combined scenarios ----

class TestCombined:
    def test_perfectly_centered_at_target_distance(self, config):
        """Target perfectly centered and at desired distance -> all zeros."""
        cmd = compute_velocity_command(
            _det(cx=0.5, cy=0.5, bh=config.target_bbox_height), config
        )
        assert cmd.forward_m_s == 0.0
        assert cmd.right_m_s == 0.0
        assert cmd.down_m_s == 0.0
        assert cmd.yawspeed_deg_s == 0.0

    def test_all_axes_active(self):
        """Target off-center in all axes simultaneously."""
        config = ControllerConfig(dead_zone_deg=0.0, dead_zone_height_percent=0.0, fixed_altitude=False)
        cmd = compute_velocity_command(
            _det(cx=0.7, cy=0.3, bh=0.15), config
        )
        assert cmd.yawspeed_deg_s > 0.0    # right -> positive yaw
        assert cmd.down_m_s < 0.0           # above center -> fly up
        assert cmd.forward_m_s > 0.0        # small bbox -> approach

    def test_custom_gains(self):
        """Custom gain values should scale the output proportionally."""
        cfg_low = ControllerConfig(
            kp_yaw=1.0, kp_down=0.04, kp_forward=1.5,
            dead_zone_deg=0.0, dead_zone_height_percent=0.0,
            fixed_altitude=False,
            max_yawspeed=9999.0, max_down_speed=9999.0,
            max_forward=9999.0, max_backward=9999.0,
        )
        cfg_high = ControllerConfig(
            kp_yaw=2.0, kp_down=0.08, kp_forward=3.0,
            dead_zone_deg=0.0, dead_zone_height_percent=0.0,
            fixed_altitude=False,
            max_yawspeed=9999.0, max_down_speed=9999.0,
            max_forward=9999.0, max_backward=9999.0,
        )
        det = _det(cx=0.65, cy=0.65, bh=0.15)
        cmd_low = compute_velocity_command(det, cfg_low)
        cmd_high = compute_velocity_command(det, cfg_high)

        assert abs(cmd_high.yawspeed_deg_s / cmd_low.yawspeed_deg_s - 2.0) < 0.01
        assert abs(cmd_high.down_m_s / cmd_low.down_m_s - 2.0) < 0.01
        assert abs(cmd_high.forward_m_s / cmd_low.forward_m_s - 2.0) < 0.01


# ---- Physics step: apply_physics_step (drone velocity -> camera-frame update) ----

class TestApplyPhysicsStep:
    """After a velocity command, camera-frame position updates (same logic as mock_control_loop)."""

    @pytest.fixture
    def step_config(self):
        return ControllerConfig(hfov=66.0, vfov=41.0)

    def test_yaw_right_moves_target_left(self, step_config):
        """Positive yawspeed (turn right) -> target moves left in frame (cx decreases)."""
        cx, cy, bh = 0.7, 0.5, 0.3
        cmd = VelocityBodyYawspeed(0.0, 0.0, 0.0, 30.0)
        dt = 0.1
        new_cx, new_cy, new_bh = apply_physics_step(cx, cy, bh, cmd, dt, step_config)
        assert new_cx < cx
        assert new_cy == cy
        assert new_bh == bh

    def test_yaw_left_moves_target_right(self, step_config):
        """Negative yawspeed (turn left) -> target moves right in frame (cx increases)."""
        cx, cy, bh = 0.3, 0.5, 0.3
        cmd = VelocityBodyYawspeed(0.0, 0.0, 0.0, -20.0)
        dt = 0.1
        new_cx, new_cy, new_bh = apply_physics_step(cx, cy, bh, cmd, dt, step_config)
        assert new_cx > cx

    def test_forward_increases_bbox_height(self, step_config):
        """Forward (approach) -> bbox height increases."""
        cx, cy, bh = 0.5, 0.5, 0.2
        cmd = VelocityBodyYawspeed(1.0, 0.0, 0.0, 0.0)
        dt = 0.1
        new_cx, new_cy, new_bh = apply_physics_step(cx, cy, bh, cmd, dt, step_config)
        assert new_bh > bh

    def test_down_moves_target_up_in_frame(self, step_config):
        """Drone descends -> target moves up in frame (cy decreases)."""
        cx, cy, bh = 0.5, 0.5, 0.3
        cmd = VelocityBodyYawspeed(0.0, 0.0, 1.0, 0.0)
        dt = 0.1
        new_cx, new_cy, new_bh = apply_physics_step(cx, cy, bh, cmd, dt, step_config)
        assert new_cy < cy
