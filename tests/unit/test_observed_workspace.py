"""Static recorded rays never become fresh dynamic observations or truth geometry."""

import numpy as np
import pytest

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


def test_constructive_ray_pyramid_covers_original_radius_despite_raster_padding():
    observed = space(2.0, 1.0)
    view = next(iter(observed.views.values()))
    view["valid_depth"][35, 40] = False
    workspace = CommonWorkspace([observed], (0.0, 0.0))
    # The large raster rectangle includes an unnecessary unknown border pixel.
    direct = workspace.query_static([[0, 0, 1]], [0.03])
    assert direct.unknown_in_envelope and direct.collision_clearance_m is None
    smaller = workspace.query_static([[0, 0, 1]], [0.0285])
    assert smaller.collision_clearance_m > 0.0015
    full = workspace.query_static([[0, 0, 1]], [0.03], cover_ids=["whole_sweep"])
    assert not full.unknown_in_envelope and full.collision_clearance_m > 0
    assert full.collision_clearance_m <= smaller.collision_clearance_m - 0.0015 + 1e-12
    assert workspace._free_covers["whole_sweep"][1] == 0.03
    assert full.source_clocks == workspace.source_clocks


@pytest.mark.parametrize("occupied", [False, True])
def test_constructive_cover_never_fills_actual_unknown_or_occupied_volume(occupied):
    observed = space(2.0, 1.0)
    view = next(iter(observed.views.values()))
    if occupied:
        view["depth_m"][40, 40] = 1.0
    else:
        view["valid_depth"][40, 40] = False
    workspace = CommonWorkspace([observed], (0.0, 0.0))
    full = workspace.query_static([[0, 0, 1]], [0.03], cover_ids=["whole_sweep"])
    assert full.collision_clearance_m is None
    assert "whole_sweep" not in workspace._free_covers


def test_subcover_certifies_whole_ball_beside_oblique_surface_and_keeps_unknown_holes():
    observed = space(2.0, 1.0)
    view = next(iter(observed.views.values()))
    u = np.arange(80)[None, :, None]
    # Measured plane z - x = 1.18, normal distance 0.18/sqrt(2) from the center.
    view["depth_m"][:] = 1.18 / (1 - (u - 40) / 100)
    workspace = CommonWorkspace([observed], (0.0, 0.0))
    direct = workspace.query_static([[0, 0, 1]], [0.1])
    assert direct.collision_clearance_m is None and not direct.occupied
    full = workspace.query_static([[0, 0, 1]], [0.1], cover_ids=["complete_ball"])
    assert full.collision_clearance_m > 0
    assert full.collision_clearance_m < 0.18 / np.sqrt(2) - 0.1
    assert workspace._free_covers["complete_ball"][1] == 0.1
    # An off-center missing ray traverses the actual sphere; subdivisions cannot fill it.
    view["valid_depth"][40, 48] = False
    unknown = CommonWorkspace([observed], (0.0, 0.0))
    assert unknown.query_static([[0, 0, 1]], [0.1], cover_ids=["hole"]).unknown_in_envelope
    assert "hole" not in unknown._free_covers


def test_moving_scaled_subcover_reuses_complete_partition_and_rejects_new_unknown(monkeypatch):
    observed = space(2.0, 1.0)
    view = next(iter(observed.views.values()))
    u = np.arange(80)[None, :, None]
    view["depth_m"][:] = 1.18 / (1 - (u - 40) / 100)
    view["valid_depth"][40, 60] = False
    workspace = CommonWorkspace([observed], (0.0, 0.0))
    first = workspace.query_static([[0, 0, 1]], [0.1], cover_ids=["ball"])
    assert first.collision_clearance_m > 0 and workspace._subcovers["ball"]
    calls = []
    original = workspace._query_uncached

    def counted(centers, radii):
        calls.append(True)
        return original(centers, radii)

    monkeypatch.setattr(workspace, "_query_uncached", counted)
    # Translation exceeds the old root margin. Retained leaves still cover the
    # WHOLE translated/scaled sphere, with each leaf paying its own growth.
    moved = workspace.query_static([[0.004, 0, 1.004]], [0.101], cover_ids=["ball"])
    assert moved.collision_clearance_m > 0 and workspace.cache_hits == 1
    assert moved.collision_clearance_m < 0.18 / np.sqrt(2) - 0.101
    assert moved.source_clocks == first.source_clocks
    reused_calls = len(calls)
    fresh = CommonWorkspace([observed], (0.0, 0.0))
    fresh_calls = []
    fresh_original = fresh._query_uncached

    def fresh_counted(centers, radii):
        fresh_calls.append(True)
        return fresh_original(centers, radii)

    monkeypatch.setattr(fresh, "_query_uncached", fresh_counted)
    assert (
        fresh.query_static([[0.004, 0, 1.004]], [0.101], cover_ids=["ball"]).collision_clearance_m
        > 0
    )
    assert reused_calls < len(fresh_calls)
    unknown = workspace.query_static([[0.15, 0, 1.15]], [0.101], cover_ids=["ball"])
    assert unknown.unknown_in_envelope and unknown.collision_clearance_m is None


def test_static_point_index_preserves_direct_occupancy_free_and_unknown_queries():
    from copy import deepcopy

    observed = space(2.0, 1.0)
    view = next(iter(observed.views.values()))
    u = np.arange(80)[None, :, None]
    view["depth_m"][:] = 1.18 / (1 - (u - 40) / 100)
    view["valid_depth"][40, 48] = False
    indexed = deepcopy(observed)
    indexed.index_static_points()
    rng = np.random.default_rng(37)
    for center, radius in zip(
        rng.uniform([-0.3, -0.2, 0.8], [0.3, 0.2, 1.4], (80, 3)),
        rng.uniform(0.001, 0.12, 80),
        strict=True,
    ):
        direct = observed.query(center[None], np.array([radius]))
        fast = indexed.query(center[None], np.array([radius]))
        assert fast == direct
    assert not next(iter(indexed.views.values()))["depth_m"].flags.writeable


def test_static_point_index_keeps_inclusive_occupancy_at_float_boundaries():
    from copy import deepcopy

    observed = space(2.0, 1.0)
    indexed = deepcopy(observed)
    indexed.index_static_points()
    for z in [np.nextafter(2.014, 0.0), 2.014, np.nextafter(2.014, np.inf)]:
        assert observed.query([[0, 0, z]], [0.01]) == indexed.query([[0, 0, z]], [0.01])


def test_coarse_stop_queries_defer_constructive_work_without_caching_unknown():
    observed = space(2.0, 1.0)
    view = next(iter(observed.views.values()))
    u = np.arange(80)[None, :, None]
    view["depth_m"][:] = 1.18 / (1 - (u - 40) / 100)
    workspace = CommonWorkspace([observed], (0.0, 0.0))
    coarse = workspace.query_static([[0, 0, 1]], [0.1], cover_ids=["box"], constructive=False)
    assert coarse.unknown_in_envelope and "box" not in workspace._free_covers
    full = workspace.query_static([[0, 0, 1]], [0.1], cover_ids=["box"])
    assert full.collision_clearance_m > 0
    reused = workspace.query_static([[0, 0, 1]], [0.1], cover_ids=["box"], constructive=False)
    assert reused.collision_clearance_m == full.collision_clearance_m
    assert reused.source_clocks == coarse.source_clocks
