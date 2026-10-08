"""The retained camera scan stays within physical neck limits."""

import copy
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
