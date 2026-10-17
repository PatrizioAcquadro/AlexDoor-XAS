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


def test_cached_free_ball_retains_volume_and_calibration_margin(monkeypatch):
    workspace = CommonWorkspace([space(2.0, 1.0)], (0.001, 0.01))
    first = workspace.query_static([[0, 0, 1]], [0.01], cover_ids=["robot_cell"])
    original = workspace._query_uncached
    calls = []

    def counted(centers, radii):
        calls.append(True)
        return original(centers, radii)

    monkeypatch.setattr(workspace, "_query_uncached", counted)
    moved = workspace.query_static([[0.001, 0, 1]], [0.011], cover_ids=["robot_cell"])
    assert not calls and workspace.cache_hits == 1
    assert moved.collision_clearance_m < first.collision_clearance_m - 0.002
    assert moved.source_clocks == first.source_clocks
    outside = workspace.query_static([[2.0, 0, 1]], [0.01], cover_ids=["robot_cell"])
    assert calls and outside.unknown_in_envelope


def test_workspace_owns_static_input_and_does_not_cache_unknown_geometry():
    rays = space(2.0, 1.0)
    workspace = CommonWorkspace([rays], (0.0, 0.0))
    rays.views.clear()
    good = workspace.query_static([[0, 0, 1]], [0.01], cover_ids=["cell"])
    assert good.collision_clearance_m > 0
    assert workspace.query_static([[0, 0, 3]], [0.01], cover_ids=["unknown"]).unknown_in_envelope
    assert "unknown" not in workspace._free_covers


def test_recorded_workspace_normalizes_explicit_legacy_sampling_without_rewriting(tmp_path):
    import json

    import h5py
    import pytest

    path = tmp_path / "workspace.hdf5"
    raw = np.array([[100.0, 0, 40], [0, 100.0, 40], [0, 0, 1]])
    with h5py.File(path, "w") as source:
        source.attrs["complete"] = True
        metadata = source.create_group("metadata")
        metadata.attrs["calibration"] = json.dumps({"depth_interval_m": [0.1, 5]})
        metadata.attrs["camera_fk_error"] = json.dumps({"position_m": 0, "rotation_rad": 0})
        observed = source.create_group("observations")
        observed["time_s"] = [1.0]
        observed["frame"] = [0]
        observed["depth_m"] = np.ones((1, 80, 80, 1)) * 2
        observed["valid_depth"] = np.ones((1, 80, 80, 1), bool)
        observed["intrinsics"] = raw[None]
        observed["camera_world"] = np.eye(4)[None]
    config = dict(
        inspection=dict(sample_times_s=[1]),
        plane_tolerance_m=0.004,
        camera_pixel_center_offset_xy=[0.5, 0.5],
    )
    common = CommonWorkspace.from_recordings([path], config, [0, 0])
    view = next(iter(common.spaces[0].views.values()))
    np.testing.assert_array_equal(view["intrinsics"][:2, 2], raw[:2, 2] - 0.5)
    assert common.source_clocks == (((1.0, 1.0),),)
    with h5py.File(path, "r+") as source:
        np.testing.assert_array_equal(source["observations/intrinsics"][0], raw)
        source["metadata"].attrs["calibration"] = json.dumps(
            {"depth_interval_m": [0.1, 5], "pixel_center_offset_xy": [0, 0]}
        )
    with pytest.raises(ValueError, match="mismatch"):
        CommonWorkspace.from_recordings([path], config, [0, 0])
