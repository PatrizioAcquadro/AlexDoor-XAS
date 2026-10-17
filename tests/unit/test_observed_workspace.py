"""Static recorded rays never become fresh dynamic observations or truth geometry."""

import numpy as np

from alexdoor_xas.perception.scan import ObservedSpace
from alexdoor_xas.perception.workspace import CommonWorkspace


def space(depth, timestamp):
    config = dict(inspection=dict(sample_times_s=[1, 2]), plane_tolerance_m=0.004)
    result = ObservedSpace(config)
    result.add(
        dict(
            time_s=np.asarray(timestamp),
            frame=np.asarray(1),
            camera_world=np.eye(4),
            intrinsics=np.array([[100.0, 0, 40], [0, 100.0, 40], [0, 0, 1]]),
            depth_m=np.full((80, 80, 1), depth),
            valid_depth=np.ones((80, 80, 1), bool),
        )
    )
    return result


def test_common_workspace_requires_all_calibration_conditions_and_keeps_source_clocks():
    a, b = space(2.0, 1.0), space(2.1, 2.0)
    common = CommonWorkspace([a, b], (0.001, 0.001))
    support = common.query_static([[0, 0, 1]], [0.01])
    assert support.collision_clearance_m > 0
    assert support.source_clocks == (((1.0, 1.0),), ((2.0, 2.0),))
    # A condition with an unobserved volume cannot be dropped in favor of another.
    blocked = CommonWorkspace([a, space(0.9, 2.0)], (0.001, 0.001))
    missing = blocked.query_static([[0, 0, 1]], [0.01])
    assert missing.unknown_in_envelope and missing.collision_clearance_m is None
    assert blocked.query_static([], []).unknown_in_envelope
    assert blocked.query_static([[0, 0, 1]], 0.01).unknown_in_envelope


def test_camera_uncertainty_erodes_static_space_without_filling_it():
    rays = space(2.0, 1.0)
    assert (
        CommonWorkspace([rays], (0.0, 0.0)).query_static([[0, 0, 1]], [0.01]).collision_clearance_m
        > 0
    )
    uncertain = CommonWorkspace([rays], (0.5, 0.5)).query_static([[0, 0, 1]], [0.01])
    assert uncertain.unknown_in_envelope and uncertain.collision_clearance_m is None
