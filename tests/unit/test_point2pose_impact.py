"""Analytic transport cases protect evaluator frame and consumer semantics."""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from alexdoor_xas.perception.diagnostics.impact import transport_impact


def test_roll_changes_orientation_without_changing_contact_normal():
    expected = np.eye(4)
    actual = np.eye(4)
    actual[:3, :3] = Rotation.from_euler("x", 30, degrees=True).as_matrix()
    actual[:3, 3] = [0.002, 0.003, 0.004]
    result = transport_impact(actual, expected)
    assert result["normal_error_deg"] == pytest.approx(0)
    assert result["rotation_error_deg"] == pytest.approx(30)
    assert result["point_signed_normal_m"] == pytest.approx(0.002)
    assert result["point_tangential_m"] == pytest.approx(0.005)
    expected[0, 0] = 1 - 1e-7
    assert transport_impact(actual, expected)["normal_error_deg"] == pytest.approx(0)


def test_target_lever_and_world_frame_invariance():
    expected, actual, target = np.eye(4), np.eye(4), np.eye(4)
    actual[:3, :3] = Rotation.from_euler("z", 90, degrees=True).as_matrix()
    target[:3, 3] = [1, 0, 0]
    result = transport_impact(actual, expected, target)
    assert result["point_error_m"] == pytest.approx(0)
    assert result["normal_error_deg"] == pytest.approx(90)
    assert result["target_position_error_m"] == pytest.approx(np.sqrt(2))
    assert result["target_rotation_error_deg"] == pytest.approx(90)
    world = np.eye(4)
    world[:3, :3] = Rotation.from_euler("xyz", [0.3, 0.6, -0.2]).as_matrix()
    world[:3, 3] = [2, -1, 3]
    rotated = transport_impact(world @ actual, world @ expected, world @ target)
    for key in result:
        assert rotated[key] == pytest.approx(result[key], abs=1e-12)


def test_missing_output_and_invalid_target_are_not_synthesized():
    assert transport_impact(None, np.eye(4)) is None
    invalid = np.full((4, 4), np.nan)
    assert transport_impact(invalid, np.eye(4)) is None
    with pytest.raises(ValueError, match="invalid_saved_target"):
        transport_impact(np.eye(4), np.eye(4), invalid)
