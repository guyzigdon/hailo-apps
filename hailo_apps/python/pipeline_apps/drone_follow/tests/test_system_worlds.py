"""System tests: deterministic world simulations with safety & tracking assertions.

Each "world" is a list of entities (people, boxes) with initial positions and drift
functions.  One entity is the tracked target.  The simulation reuses
compute_velocity_command + apply_physics_step to step through time, then checks:

  1. Safety  — no person's bbox_height ever reaches 0.9 (collision proxy)
  2. Tracking — after warm-up the controller centres and holds the target
"""

import math
import time
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

import pytest

from drone_control import (
    ControllerConfig,
    Detection,
    apply_physics_step,
    compute_velocity_command,
)

# --------------------------------------------------------------------------
# Simulation engine
# ---------------------------------------------------------------------------

@dataclass
class Entity:
    label: str                          # "person" or "box"
    cx: float                           # centre-x  (0-1)
    cy: float                           # centre-y  (0-1)
    bh: float                           # bbox height (0-1)
    drift: Callable[[float], Tuple[float, float, float]] = field(
        default_factory=lambda: (lambda _t: (0.0, 0.0, 0.0))
    )


@dataclass
class Snapshot:
    """One time-step record for every entity."""
    entities: List[Tuple[str, float, float, float]]   # [(label, cx, cy, bh), ...]


def run_world(
    entities: List[Entity],
    tracked_index: int,
    config: ControllerConfig,
    num_steps: int = 500,
    dt: float = 0.1,
) -> List[Snapshot]:
    """Step through a multi-entity world, return per-step snapshots."""
    snapshots: List[Snapshot] = []

    for step in range(num_steps):
        t = step * dt

        # 1. Record snapshot
        snap = Snapshot(
            entities=[(e.label, e.cx, e.cy, e.bh) for e in entities]
        )
        snapshots.append(snap)

        # 2. Build detection from tracked entity
        te = entities[tracked_index]
        det = Detection(
            label=te.label,
            confidence=0.99,
            center_x=te.cx,
            center_y=te.cy,
            bbox_height=te.bh,
            timestamp=time.monotonic(),
        )

        # 3. Compute velocity command
        cmd = compute_velocity_command(det, config)

        # 4. Apply physics (drone motion) to every entity
        for e in entities:
            e.cx, e.cy, e.bh = apply_physics_step(
                e.cx, e.cy, e.bh, cmd, dt, config
            )

        # 5. Apply per-entity drift, clamp to bounds
        for e in entities:
            dx, dy, dbh = e.drift(t)
            e.cx = max(0.01, min(0.99, e.cx + dx))
            e.cy = max(0.01, min(0.99, e.cy + dy))
            e.bh = max(0.02, min(0.95, e.bh + dbh))

    return snapshots

# ---------------------------------------------------------------------------
# World catalogue
# ---------------------------------------------------------------------------

def _no_drift(_t: float) -> Tuple[float, float, float]:
    return (0.0, 0.0, 0.0)


def _circle_drift(t: float) -> Tuple[float, float, float]:
    dx = 0.02 * math.cos(2 * math.pi * t / 10.0)
    dbh = 0.008 * math.sin(2 * math.pi * t / 10.0)
    return (dx, 0.0, dbh)


def _approach_drift(_t: float) -> Tuple[float, float, float]:
    """Person walks toward drone (bh increases)."""
    return (0.0, 0.0, 0.003)


def _retreat_drift(_t: float) -> Tuple[float, float, float]:
    """Person walks away from drone (bh decreases)."""
    return (0.0, 0.0, -0.003)


def _fast_approach_drift(_t: float) -> Tuple[float, float, float]:
    """Person jogs toward drone (bh increases quickly)."""
    return (0.0, 0.0, 0.008)


def _sweep_drift(t: float) -> Tuple[float, float, float]:
    dx = 0.03 * math.sin(2 * math.pi * t / 5.0)
    return (dx, 0.0, 0.0)


def _default_config() -> ControllerConfig:
    return ControllerConfig(
        target_bbox_height=0.3,
        kp_yaw=2.0,
        kp_forward=3.0,
        max_bbox_height_safety=0.8,
    )


def world_solo_static():
    return (
        [Entity("person", cx=0.7, cy=0.5, bh=0.15, drift=_no_drift)],
        0,
    )


def world_solo_circle():
    return (
        [Entity("person", cx=0.6, cy=0.5, bh=0.20, drift=_circle_drift)],
        0,
    )


def world_solo_approach():
    return (
        [Entity("person", cx=0.5, cy=0.5, bh=0.25, drift=_approach_drift)],
        0,
    )


def world_solo_retreat():
    return (
        [Entity("person", cx=0.5, cy=0.5, bh=0.35, drift=_retreat_drift)],
        0,
    )


def world_solo_fast_approach():
    return (
        [Entity("person", cx=0.5, cy=0.5, bh=0.20, drift=_fast_approach_drift)],
        0,
    )


def world_duo_spread():
    """Two people: target at left, bystander sweeps right side."""
    return (
        [
            Entity("person", cx=0.3, cy=0.5, bh=0.20, drift=_no_drift),
            Entity("person", cx=0.7, cy=0.5, bh=0.25, drift=_sweep_drift),
        ],
        0,  # track the left person
    )


def world_duo_staggered():
    """Two people at different distances: track the far one, closer bystander stays safe."""
    return (
        [
            Entity("person", cx=0.5, cy=0.5, bh=0.15, drift=_no_drift),   # far (tracked)
            Entity("person", cx=0.4, cy=0.5, bh=0.45, drift=_no_drift),   # closer bystander
        ],
        0,  # track the far person
    )


def world_person_and_box():
    """One person circling + one static box obstacle."""
    return (
        [
            Entity("person", cx=0.5, cy=0.5, bh=0.20, drift=_circle_drift),
            Entity("box", cx=0.6, cy=0.5, bh=0.30, drift=_no_drift),
        ],
        0,  # track the person
    )


ALL_WORLDS = {
    "solo_static": world_solo_static,
    "solo_circle": world_solo_circle,
    "solo_approach": world_solo_approach,
    "solo_retreat": world_solo_retreat,
    "solo_fast_approach": world_solo_fast_approach,
    "duo_spread": world_duo_spread,
    "duo_staggered": world_duo_staggered,
    "person_and_box": world_person_and_box,
}

WORLD_NAMES = list(ALL_WORLDS.keys())

# Worlds where distance tracking assertion applies (exclude solo_fast_approach)
DISTANCE_TRACKING_WORLDS = [w for w in WORLD_NAMES if w != "solo_fast_approach"]

# Simulation parameters
NUM_STEPS = 500
DT = 0.1
WARMUP_STEPS = 50

# ---------------------------------------------------------------------------
# Assertion thresholds
# ---------------------------------------------------------------------------
SAFETY_BH_LIMIT = 0.9        # person fills 90% of frame ~ contact
HORIZONTAL_TOLERANCE = 0.12   # avg |cx - 0.5| after warm-up
DISTANCE_TOLERANCE = 0.20     # avg |bh - target| after warm-up


# ---------------------------------------------------------------------------
# Test classes
# ---------------------------------------------------------------------------

class TestNoPersonCollision:
    """For every person entity, at every step, bh < SAFETY_BH_LIMIT."""

    @pytest.mark.parametrize("world_name", WORLD_NAMES)
    def test_safety(self, world_name):
        entities, tracked_idx = ALL_WORLDS[world_name]()
        config = _default_config()
        snapshots = run_world(entities, tracked_idx, config, NUM_STEPS, DT)

        for step_i, snap in enumerate(snapshots):
            for label, cx, cy, bh in snap.entities:
                if label == "person":
                    assert bh < SAFETY_BH_LIMIT, (
                        f"[{world_name}] step {step_i}: person bh={bh:.3f} "
                        f">= {SAFETY_BH_LIMIT} (collision)"
                    )


class TestTrackingQuality:
    """After warm-up, the controller converges on the target."""

    @pytest.mark.parametrize("world_name", WORLD_NAMES)
    def test_horizontal_tracking(self, world_name):
        """Average |cx - 0.5| of tracked person stays below threshold."""
        entities, tracked_idx = ALL_WORLDS[world_name]()
        config = _default_config()
        snapshots = run_world(entities, tracked_idx, config, NUM_STEPS, DT)

        post_warmup = snapshots[WARMUP_STEPS:]
        cx_errors = [
            abs(snap.entities[tracked_idx][1] - 0.5) for snap in post_warmup
        ]
        avg_cx_error = sum(cx_errors) / len(cx_errors)

        assert avg_cx_error < HORIZONTAL_TOLERANCE, (
            f"[{world_name}] avg horizontal error {avg_cx_error:.4f} "
            f">= {HORIZONTAL_TOLERANCE}"
        )

    @pytest.mark.parametrize("world_name", DISTANCE_TRACKING_WORLDS)
    def test_distance_tracking(self, world_name):
        """Average |bh - target| of tracked person stays below threshold."""
        entities, tracked_idx = ALL_WORLDS[world_name]()
        config = _default_config()
        snapshots = run_world(entities, tracked_idx, config, NUM_STEPS, DT)

        target_bh = config.target_bbox_height
        post_warmup = snapshots[WARMUP_STEPS:]
        bh_errors = [
            abs(snap.entities[tracked_idx][3] - target_bh) for snap in post_warmup
        ]
        avg_bh_error = sum(bh_errors) / len(bh_errors)

        assert avg_bh_error < DISTANCE_TOLERANCE, (
            f"[{world_name}] avg distance error {avg_bh_error:.4f} "
            f">= {DISTANCE_TOLERANCE}"
        )
