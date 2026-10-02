"""Essential causal material/selection regressions; no physical or release claims."""

from dataclasses import replace

import numpy as np
import pytest

from alexdoor_xas.action.frames import ObjectFrame, rot_z
from alexdoor_xas.perception.contracts import (
    DoorEstimate,
    FieldSupport,
    LocalContactSelection,
    LocalMaterialState,
    LocalPatchState,
    validate_local_contact,
    validate_local_transition,
)
from alexdoor_xas.perception.material import MaterialTracker, motion_bounds
from alexdoor_xas.perception.provider import CueEngine, GeometryProvider
from test_geometric_perception import EmptyWorker, recipe, sensor
from test_scan_fusion import measured_surface


def supported_patch(patch_id, time=1.0):
    support = FieldSupport(0, time, time, 0, 0.004, 0.02)
    return LocalPatchState(
        patch_id,
        ObjectFrame(np.array([1, 0, 1.1]), np.eye(3)),
        support,
        support,
        support,
        (0.04, 0.04),
        "tracked_material",
    )


def test_visual_reference_and_explicit_contact_have_independent_lifetimes():
    reference, contact = supported_patch("visual"), supported_patch("contact", 0.5)
    selection = LocalContactSelection("s1", "contact", 0.5, "diagnostic")
    state = LocalMaterialState(0, "visual", (reference, contact), selection)
    with pytest.raises(ValueError, match="stale_dynamic_support"):
        validate_local_contact(state, 1)
    # A stale visual reference does not erase a separately supported selected patch.
    state = replace(state, patches=(supported_patch("visual", 0.5), supported_patch("contact")))
    assert validate_local_contact(state, 1).patch_id == "contact"
    assert not DoorEstimate(1, False, "local", local=state).valid


def test_selection_requires_source_identity_time_and_explicit_predecessor():
    previous = LocalContactSelection("s1", "high", 1, "diagnostic")
    validate_local_transition(None, previous)
    with pytest.raises(ValueError, match="reselected"):
        validate_local_transition(previous, replace(previous, patch_id="low"))
    with pytest.raises(ValueError, match="missing_contact_transition"):
        validate_local_transition(previous, LocalContactSelection("s2", "low", 2, "diagnostic"))
    following = LocalContactSelection("s2", "low", 2, "diagnostic", "s1")
    validate_local_transition(previous, following)
    state = LocalMaterialState(0, "high", (supported_patch("low", 2),), following)
    assert validate_local_contact(state, 2).patch_id == "low"
    with pytest.raises(ValueError, match="missing_explicit"):
        validate_local_contact(replace(state, selection=replace(following, source="tracker")), 2)


def test_material_initialization_uses_available_local_membership_not_whole_plane():
    surface, sample, _ = measured_surface()
    covers = tuple(
        np.array([[0, y, -0.01], [0, y + 0.01, -0.01], [0, y + 0.01, 0.01], [0, y, 0.01]])
        for y in (-0.03, 0.02)
    )
    tracker = MaterialTracker(recipe().config, 0, covers)
    tracker.request("local", [99, 0, 1], reference=True)
    tracker.observe([surface], sample, 0.2)
    track = tracker.tracks["local"]
    assert track.geometry.available_s == 0.2
    assert track.geometry.acquired_s == 0
    assert np.ptp(track.reference.points[:, 1]) < 0.17
    with pytest.raises(ValueError, match="unavailable_field_support"):
        tracker.select(LocalContactSelection("s", "local", 0.1, "diagnostic"), 0.1)
    tracker.select(LocalContactSelection("s", "local", 0.2, "diagnostic"), 0.2)
    assert tracker.selection.patch_id == "local"
    assert tracker.reference_id == "local"
    assert not tracker.rigid_links


def test_local_path_preserves_static_scene_support_without_refreshing_material():
    _, sample, cue = measured_surface()
    provider = GeometryProvider(recipe(), CueEngine(EmptyWorker(), replay=True))
    tracker = provider.configure_material(())
    tracker.request("unobserved", [1, 5, 1], reference=True)
    provider._fuse(sample, 8, dict(cue, latency_s=0), available_s=0.1)
    assert provider.scan_memory.surfaces
    assert provider.scan_state.unresolved_surfaces
    assert provider.scan_memory.surfaces[0].observations[0].available_s == 0.1
    assert tracker.tracks["unobserved"].geometry is None
    assert tracker.tracks["unobserved"].dynamic is None


def test_material_mask_excludes_other_coplanar_texture_during_seed_replenishment_and_loss():
    from alexdoor_xas.perception.tracking import PixelMotionTracker

    surface, sample, _ = measured_surface()
    sample["rgb"] = np.random.default_rng(18).integers(0, 256, sample["rgb"].shape, dtype=np.uint8)
    mask = np.zeros(sample["rgb"].shape[:2], bool)
    mask[80:160, 80:150] = True
    tracker = PixelMotionTracker()
    tracker.update(sample, surface, material_mask=mask)
    tracker.pixels = tracker.pixels[:30]
    tracker.source = tracker.source[:30]
    tracker.source_uncertainty = tracker.source_uncertainty[:30]
    assert tracker.update(sample, surface, material_mask=mask) is not None
    assert tracker.diagnostics["pixel_reseed"] and len(tracker.pixels) > 30
    xy = np.rint(tracker.pixels).astype(int)
    assert mask[xy[:, 1], xy[:, 0]].all()
    # Texture elsewhere on the same plane cannot keep this material reference alive.
    assert tracker.update(sample, surface, material_mask=~mask) is None
    assert tracker.gray is None


@pytest.mark.parametrize("sign", [-1, 1])
def test_static_coplanarity_never_transfers_motion_and_verified_link_propagates_bounds(sign):
    tracker = MaterialTracker(recipe().config, 0, ())
    for name, height in (("visual", 1.5), ("contact", 1.1)):
        tracker.request(name, [1, 0, height], reference=name == "visual")
        track = tracker.tracks[name]
        patch = supported_patch(name)
        track.geometry = patch.geometry_support
        track.closed_pose = ObjectFrame(np.array([1, 0, height]), np.eye(3))
        track.pose = track.closed_pose
        track.identity = track.dynamic = patch.pose_support
        track.clearance = (0.1, 0.1)
        # Isolate the association consumer from the optical tracker.
        track.update = lambda sensor: None
    visual, contact = tracker.tracks.values()
    contact.geometry = replace(contact.geometry, position_bound_m=0.03, rotation_bound_rad=0.1)
    contact.dynamic = contact.identity = replace(
        contact.geometry, position_bound_m=0.034, rotation_bound_rad=0.12
    )
    visual.observed = True
    contact.observed = False
    visual.motion = (np.eye(3), np.zeros(3), 0)
    tracker.update(sensor(1, 1))
    assert contact.reason == "acquiring_material" and not tracker.rigid_links
    contact.observed = True
    for _ in range(3):
        visual.motion = contact.motion = (rot_z(sign * 0.2), np.array([0.1, 0, 0]), 0.001)
        tracker.update(sensor(1, 1))
    assert tracker.rigid_links == {("contact", "visual")}
    original_geometry = contact.geometry
    contact.observed = False
    tracker.update(sensor(1, 1))
    assert contact.reason == "verified_rigid_transfer"
    assert contact.geometry is original_geometry
    assert contact.dynamic.position_bound_m > visual.dynamic.position_bound_m
    assert contact.dynamic.position_bound_m >= (
        contact.geometry.position_bound_m + visual.dynamic.position_bound_m
    )
    assert contact.dynamic.rotation_bound_rad >= (
        contact.geometry.rotation_bound_rad + visual.dynamic.rotation_bound_rad
    )
    contact.observed = True
    contact.motion = (rot_z(-sign * 0.2), np.zeros(3), 0.001)
    tracker.update(sensor(1, 1))
    assert not tracker.rigid_links


def test_supported_axis_represents_the_exact_explicit_contact_instead_of_the_visual_reference():
    visual = replace(
        supported_patch("visual"), world_pose=ObjectFrame(np.array([1, -0.1, 1.5]), np.eye(3))
    )
    contact = supported_patch("contact")
    state = LocalMaterialState(
        0, "visual", (visual, contact), LocalContactSelection("s", "contact", 1, "diagnostic")
    )
    panel = ObjectFrame(np.array([0, 0.5, 0]), rot_z(0.2))
    converted = MaterialTracker(recipe().config, 0, ()).operational_contact(state, panel, 1)
    assert converted.patch_id == "contact"
    np.testing.assert_array_equal(converted.world_pose.origin, contact.world_pose.origin)
    np.testing.assert_allclose(
        panel.point_to_world(converted.local_pose.origin), contact.world_pose.origin, atol=1e-12
    )
    np.testing.assert_allclose(
        panel.rot @ converted.local_pose.rot, contact.world_pose.rot, atol=1e-12
    )


def test_local_generation_rejects_boolean_episode_identity():
    state = LocalMaterialState(
        False,
        "contact",
        (supported_patch("contact"),),
        LocalContactSelection("s", "contact", 1, "diagnostic"),
    )
    with pytest.raises(ValueError, match="wrong_episode_generation"):
        validate_local_contact(state, 1, generation=0)


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


def test_field_gates_allow_roundoff_without_an_extra_tick_of_dynamic_freshness():
    from alexdoor_xas.perception.contracts import POSITION_LIMIT_M, ROTATION_LIMIT_RAD

    support = FieldSupport(0, 0, 0, 0, POSITION_LIMIT_M + 1e-7, ROTATION_LIMIT_RAD + 1e-7)
    support.require(0.15 + 1e-12, 0, dynamic=True)
    support.require_bounds(position=True, rotation=True, qualification=True)
    with pytest.raises(ValueError, match="stale_dynamic_support"):
        support.require(0.15 + 1 / 60, 0, dynamic=True)
    with pytest.raises(ValueError, match="excessive"):
        replace(support, position_bound_m=0.011).require_bounds(position=True, qualification=True)


def test_rigid_uncertainty_increases_with_depth_error_and_poor_support_span():
    rng = np.random.default_rng(18)
    cloud = rng.normal(size=(40, 3))
    p, r = motion_bounds(cloud, 0.001, 0.002)
    p2, r2 = motion_bounds(cloud * 0.01, 0.001, 0.004)
    assert p2 > p and r2 > r


def test_request_cancellation_preserves_episode_static_support_and_reset_clears_it():
    provider = GeometryProvider(recipe(), CueEngine(EmptyWorker(), replay=True))
    material = provider.configure_material(())
    material.request("visual", [1, 0, 1], reference=True)
    provider.update(sensor(0, 8))
    generation = provider.generation
    provider.update(sensor(0.3, 26))
    assert provider.generation == generation
    assert provider.material is material and provider.scan_memory.generation == generation
    provider.reset()
    assert provider.generation == generation + 1 and provider.material is None


def test_supplied_late_completion_and_reset_never_backdate_or_cross_generations():
    class Worker(EmptyWorker):
        def infer(self, rgb):
            return dict(super().infer(rgb), available_s=4.3)

    replay, live = CueEngine(Worker(), replay=True), CueEngine(Worker(), replay=False)
    try:
        for engine in (replay, live):
            engine.submit(sensor(), 0)
        live.pending[1].result(timeout=1)
        assert replay.poll(4.2) is live.poll(4.2) is None
        assert replay.poll(4.3)[3] == live.poll(4.3)[3] == 4.3
        for engine in (replay, live):
            engine.submit(sensor(), 0)
            engine.reset()
        live.pending[1].result(timeout=1)
        assert replay.poll(4.3) is live.poll(4.3) is None
    finally:
        live.close()


def test_observed_hinge_is_retained_without_refreshing_dynamic_support_on_loss():
    from alexdoor_xas.perception.geometry import Surface

    tracker = MaterialTracker(recipe().config, 0, ())
    tracker.request("visual", [1, 0, 1], reference=True)
    track = tracker.tracks["visual"]
    patch = supported_patch("visual")
    track.geometry = track.dynamic = track.identity = patch.pose_support
    track.reference = Surface(
        np.array([[1, 0, 1], [1, 0.1, 1], [1, 0, 1.1], [1, 0.1, 1.1]]),
        np.array([1.0, 0, 0]),
        1.0,
        np.ones(384),
        np.empty((0, 3)),
        np.empty((0, 384)),
        0.001,
        1.0,
    )
    track.pose = track.closed_pose = patch.world_pose
    track.observed = True
    hinge = np.array([0.0, 0.4, 0.0])
    track.motions = [(rot_z(a), hinge - rot_z(a) @ hinge, 0.001) for a in (0.2, 0.3, 0.4)]
    track.motion = track.motions[-1]
    before = tracker.state(1)
    assert len(before.hypotheses) == 1 and before.signed_angle == pytest.approx(0.4)
    track.observed = False
    after = tracker.state(2)
    assert after.hypotheses is before.hypotheses
    assert after.angle_support.supported_s == 1
    with pytest.raises(ValueError, match="stale_dynamic_support"):
        after.angle_support.require(2, 0, dynamic=True)


def test_live_revisit_uses_existing_views_speed_and_two_second_holds():
    from alexdoor_xas.perception.material_live import reacquisition_schedule

    inspection = recipe().config["inspection"]
    points = reacquisition_schedule(inspection, 1 / 60)
    speed = abs(np.diff(points[:, 1:], axis=0) / np.diff(points[:, :1], axis=0))
    assert speed.max() <= inspection["max_neck_speed_rad_s"] + 1e-10
    np.testing.assert_allclose(points[-4, 1:], [-0.6, -0.48])
    np.testing.assert_allclose(points[-1, 1:], inspection["waypoints"][-1][1:])
    assert points[-3, 0] - points[-4, 0] == 2 and points[-1, 0] - points[-2, 0] == 2


def test_visibility_audit_distinguishes_occluded_region_from_material_reacquisition():
    from alexdoor_xas.perception.material_live import depth_visibility

    _, sample, _ = measured_surface()
    candidate = dict(position=[1, 0, 1], rotation=np.eye(3))
    visible = depth_visibility(sample, candidate, 0.01)
    assert visible["depth_consistent"] == visible["samples"]
    assert not visible["material_identity_verified"]
    # A closer occluder leaves the saved region in view but removes its depth support.
    sample["depth_m"][:] = 0.5
    occluded = depth_visibility(sample, candidate, 0.01)
    assert occluded["in_view"] == visible["samples"] and not occluded["depth_consistent"]


def test_local_replay_live_identical_observations_and_completion_events(monkeypatch):
    import json
    from dataclasses import asdict

    from alexdoor_xas.perception.evaluation import json_safe
    from alexdoor_xas.perception.provider import PrototypeRecipe

    monkeypatch.setattr("alexdoor_xas.perception.provider.time.perf_counter", lambda: 0.0)
    surface, sample, cue = measured_surface()
    sample["rgb"] = np.random.default_rng(7).integers(1, 256, sample["rgb"].shape, dtype=np.uint8)
    covers = tuple(
        np.array([[0, y, -0.01], [0, y + 0.01, -0.01], [0, y + 0.01, 0.01], [0, y, 0.01]])
        for y in (-0.03, 0.02)
    )
    config = dict(recipe().config, material_radius_m=0.3)
    binding = PrototypeRecipe(json.dumps(config))

    class Worker:
        def infer(self, rgb):
            return dict(cue, latency_s=0.05)

    providers = [
        GeometryProvider(binding, CueEngine(Worker(), replay=replay)) for replay in (True, False)
    ]
    try:
        for provider in providers:
            local = provider.configure_material(covers)
            local.request("visual", [1, 0, 1.4], reference=True)
            local.request("contact", [1, 0, 1.1])
        for row, t in enumerate((0, 0.05, 0.1, 0.15, 0.2, 0.25)):
            observation = dict(sample, time_s=t, frame=row + 8)
            results = []
            for provider in providers:
                if provider.engine.pending is not None and not provider.engine.replay:
                    provider.engine.pending[1].result(timeout=1)
                estimate = provider.update(observation)
                results.append(
                    json.dumps(
                        json_safe(asdict(estimate.local) if estimate.local is not None else None),
                        sort_keys=True,
                    )
                )
            assert results[0] == results[1]
    finally:
        for provider in providers:
            provider.engine.close()
