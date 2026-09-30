"""The retained camera scan stays within physical neck limits."""

import copy
from pathlib import Path

import pytest

from alexdoor_xas.perception.inspection import load_inspection, validate_inspection


def test_common_scan_and_unsafe_changes():
    config = load_inspection(
        Path(__file__).resolve().parents[1] / "configs/perception_inspection.json"
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
