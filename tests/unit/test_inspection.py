"""The retained camera scan stays within physical neck limits."""

import copy
import json
from pathlib import Path

import pytest

from alexdoor_xas.perception.inspection import load_inspection, validate_inspection


def test_common_scan_and_unsafe_changes():
    config = load_inspection(
        Path(__file__).resolve().parents[2] / "configs/perception_inspection.json"
    )
    for waypoints, samples in [
        ([], [1]),
        ([[0, 0, 0], [1, 2, 0]], [1]),
        ([[0, 0, 0], [1, 0, 0]], []),
        ([[0, 0, 0], [1, 0, 0]], [float("nan")]),
    ]:
        changed = copy.deepcopy(config)
        changed.update(waypoints=waypoints, sample_times_s=samples)
        with pytest.raises(ValueError):
            validate_inspection(changed)


def test_inspection_guard_accepts_sample_resolution_but_rejects_meaningful_excess():
    from alexdoor_xas.perception.inspection import (
        inspection_tolerances,
        inspection_within_limits,
    )

    dt, speed = 1 / 60, 0.4
    tolerance = inspection_tolerances(dt, speed)
    assert json.loads(json.dumps(tolerance)) == tolerance
    metrics = dict(tool_drift_m=0.000057, door_angle_rad=0.000123, neck_error_rad=0.100087)
    assert inspection_within_limits(metrics, dt, speed)
    metrics["neck_error_rad"] = 0.1 + tolerance["neck_error_rad"] + 1e-5
    assert not inspection_within_limits(metrics, dt, speed)
    metrics.update(neck_error_rad=0.1, tool_drift_m=0.01 + 1e-7)
    assert inspection_within_limits(metrics, dt, speed)
    metrics["tool_drift_m"] = 0.011
    assert not inspection_within_limits(metrics, dt, speed)
    metrics["tool_drift_m"] = float("nan")
    assert not inspection_within_limits(metrics, dt, speed)


@pytest.mark.parametrize("initial_angle", [0.0, 1.48])
def test_teacher_and_io_scan_hold_the_same_joint_anchor(initial_angle):
    from types import SimpleNamespace

    import numpy as np

    from alexdoor_xas.perception.inspection import run_inspection
    from alexdoor_xas.policies.purdue import PurdueIO

    q0 = np.linspace(-0.2, 0.2, 7)
    targets = []
    joints = np.r_[q0, 0, 0][None]
    env = SimpleNamespace(
        step_dt=0.1,
        arm_ids=list(range(7)),
        neck_ids=[7, 8],
        robot=SimpleNamespace(data=SimpleNamespace(joint_pos=joints)),
        door=SimpleNamespace(data=SimpleNamespace(joint_pos=np.full((1, 1), initial_angle))),
        capture=SimpleNamespace(sample=SimpleNamespace(joint_position=joints)),
        contact_history=[dict(forbidden=False)],
        tool_pose=lambda: (np.zeros((1, 3)), np.array([[0, 0, 0, 1]])),
    )

    def step(delta):
        targets.append(joints[0, :7] + delta)
        joints[0, :7] += delta
        joints[0, 0] += 0.0001

    env.step_a1 = step
    env.set_neck_target = lambda neck: joints.__setitem__((0, slice(7, 9)), neck)
    report = run_inspection(
        env, dict(waypoints=[[0, 0, 0], [0.3, 0.02, 0]], max_neck_speed_rad_s=0.4)
    )
    assert report["initial_door_angle_rad"] == initial_angle
    assert report["door_angle_rad"] == 0.0
    io = PurdueIO.__new__(PurdueIO)
    io.env, io.parked_joints = env, q0.copy()
    io.hold_step()
    np.testing.assert_allclose(targets, np.broadcast_to(q0, (4, 7)), atol=1e-15)


@pytest.mark.parametrize("field,value", [("mount_pitch_rad", 0), ("max_neck_speed_rad_s", 0.5)])
def test_observation_schedule_cannot_recalibrate_mount_or_speed(field, value):
    from alexdoor_xas.perception.inspection import acquisition_inspection

    original = load_inspection(
        Path(__file__).resolve().parents[2] / "configs/perception_inspection.json"
    )
    changed = copy.deepcopy(original)
    changed[field] = value
    with pytest.raises(ValueError, match="cannot change inspection calibration"):
        acquisition_inspection(original, changed)


def test_observation_schedule_retains_physical_neck_limits():
    from alexdoor_xas.perception.inspection import acquisition_inspection

    original = load_inspection(
        Path(__file__).resolve().parents[2] / "configs/perception_inspection.json"
    )
    changed = copy.deepcopy(original)
    changed["waypoints"][-1][1] = 2.0
    with pytest.raises(ValueError, match="neck"):
        acquisition_inspection(original, changed)
