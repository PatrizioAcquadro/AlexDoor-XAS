"""Essential causal, metric and complete-state behavior without model execution."""

from dataclasses import replace

import numpy as np
import pytest

from alexdoor_xas.action.frames import ObjectFrame, rot_z
from alexdoor_xas.perception.contracts import DoorEstimate, validate_complete
from alexdoor_xas.perception.control import ObservedControlChecks
from alexdoor_xas.perception.evaluation import (
    COMPONENTS,
    LIMITS,
    quantiles,
    score,
    state_errors,
)
from alexdoor_xas.perception.geometry import (
    Surface,
    contact_frame,
    deproject,
    dimensions_supported,
    extent_edges,
    plane_fit,
    project,
)
from alexdoor_xas.perception.provider import CueEngine, GeometryProvider
from alexdoor_xas.policies.common.b1_contract import PerceptionBinding
from alexdoor_xas.policies.observations import require_estimate
from alexdoor_xas.recording.b1 import OBS_KEYS
from perception_helpers import EmptyWorker, recipe, sensor


def complete_state(angle=0.3):
    hinge = ObjectFrame(np.array([0.1, 0.4, 0.0]), np.eye(3))
    rotation = rot_z(angle)
    local = np.array([-0.02, -0.25, 1.09])
    return DoorEstimate(
        0.0,
        True,
        "observed",
        0.9,
        hinge,
        rotation,
        angle,
        np.array([0.9, 2.0, 0.04]),
        hinge.origin + rotation @ local,
        rotation,
    )


def test_optical_axis_depth_roundtrip_in_moving_calibrated_camera():
    camera = np.eye(4)
    camera[:3, :3] = rot_z(0.7)
    camera[:3, 3] = [0.1, -0.2, 1.3]
    intrinsics = np.array([[600, 0, 320], [0, 590, 240], [0, 0, 1]])
    pixels, depth = np.array([[2, 4], [300, 220], [630, 470]]), np.array([0.8, 1.1, 2.2])
    world = deproject(depth, pixels, intrinsics, camera)
    observed, z = project(world, intrinsics, camera)
    np.testing.assert_allclose(observed, pixels, atol=1e-9)
    np.testing.assert_allclose(z, depth, atol=1e-9)


def test_robust_surface_fit_rejects_nonpanel_depth():
    rng = np.random.default_rng(18)
    plane = np.c_[
        rng.normal(0.2, 0.0005, 600), rng.uniform(-0.5, 0.5, 600), rng.uniform(0.1, 2, 600)
    ]
    points = np.r_[plane, rng.uniform(-2, 2, (250, 3))]
    normal, offset, support, residual = plane_fit(points, 0.004, 120, rng)
    assert abs(normal[0]) > 0.999 and abs(abs(offset) - 0.2) < 0.001
    assert support[:600].mean() > 0.99 and support[600:].mean() < 0.05
    assert residual < 0.002


def test_clipped_extent_is_not_a_measured_dimension():
    rng = np.random.default_rng(8)
    points = np.c_[np.ones(800), rng.uniform(-0.5, 0.5, 800), rng.uniform(0.1, 2, 800)]
    s = Surface(
        points,
        np.array([1.0, 0, 0]),
        1.0,
        np.ones(384) / np.sqrt(384),
        np.empty((0, 3)),
        np.empty((0, 384)),
        0.001,
        1.0,
        {0, 1},
    )
    assert not dimensions_supported(s, 0.01)
    sample = sensor()
    sample["camera_world"][:3, :3] = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    # The entire observed plane fills the image: no silhouette can certify its extents.
    front = deproject(
        np.ones(256),
        np.c_[np.tile(np.arange(16), 16), np.repeat(np.arange(16), 16)],
        sample["intrinsics"],
        sample["camera_world"],
    )
    assert not extent_edges(
        front, np.array([1.0, 0.0, 0.0]), np.ones((16, 16), bool), sample, 0.004
    )
    normal = np.array([1.0, 0.01, 0.1])
    np.testing.assert_allclose(contact_frame(normal)[:, 0], normal / np.linalg.norm(normal))


def test_reset_discards_public_estimate_and_unsafe_contact_never_reaches_io():
    from types import SimpleNamespace

    from alexdoor_xas.policies.purdue import PurdueIO

    provider = GeometryProvider(recipe(), CueEngine(EmptyWorker(), replay=True))
    provider.last_estimate = complete_state()
    provider.reset()
    assert provider.last_estimate is None and provider.encoding is None
    io = PurdueIO.__new__(PurdueIO)
    io.safety = SimpleNamespace(before_command=lambda stage: "force_feedback_unavailable")
    # No environment exists: rejecting a command must happen before any simulator access.
    with pytest.raises(RuntimeError, match="force_feedback_unavailable"):
        io.execute(SimpleNamespace(stage="contact"))


def test_complete_state_cannot_be_replaced_by_correct_composed_contact():
    estimate = complete_state()
    validate_complete(estimate)
    for bad in (
        replace(estimate, dimensions=None),
        replace(estimate, panel_rotation=rot_z(1)),
        replace(estimate, contact_rotation=np.zeros((3, 3))),
    ):
        with pytest.raises(ValueError):
            require_estimate(bad, 0, 0.15)
    truth = dict(
        hinge_origin=estimate.frame.origin,
        hinge_rotation=np.eye(3),
        dimensions=estimate.dimensions,
        signed_angle=estimate.signed_angle,
        contact_position=estimate.contact_position,
        contact_rotation=estimate.contact_rotation,
        contact_local=np.array([-0.02, -0.25, 1.09]),
        contact_rotation_local=np.eye(3),
    )
    np.testing.assert_allclose(state_errors(estimate, truth), 0, atol=1e-12)
    wrong = replace(estimate, frame=ObjectFrame(estimate.frame.origin + [1, 0, 0], np.eye(3)))
    errors = state_errors(wrong, truth)
    assert errors[COMPONENTS.index("hinge_origin_m")] == pytest.approx(1)
    assert errors[COMPONENTS.index("contact_position_m")] == pytest.approx(0)


def test_missing_and_rejected_states_stay_in_gate_denominator():
    errors = np.zeros((100, len(COMPONENTS)))
    accepted = np.ones(100, bool)
    accepted[:6] = False
    errors[:6] = np.nan
    report = score(errors, accepted, np.full(100, "missing"), np.ones(100, bool))
    assert report["valid_coverage"] == 0.94 and not report["offline_passed"]
    accepted[:] = True
    errors[:] = 0
    errors[:6, 0] = 1
    report = score(errors, accepted, np.full(100, "observed"), np.ones(100, bool))
    assert report["accepted_state_precision"] == 0.94 and not report["offline_passed"]
    report = score(errors, np.zeros(100, bool), np.full(100, "missing"), np.ones(100, bool))
    assert report["accepted_state_precision"] == 0 and not report["offline_passed"]
    assert set(report["errors_accepted"]) >= set(LIMITS)


def test_completion_order_reset_and_no_backdating():
    engine = CueEngine(EmptyWorker(), replay=True)
    assert engine.submit(sensor(), 0)
    assert not engine.submit(sensor(4.01, 249), 0)
    assert engine.poll(4.049) is None
    event = engine.poll(4.05)
    assert int(event[0]["frame"]) == 248 and float(event[0]["time_s"]) == 4
    assert engine.submit(sensor(7, 428), 1)
    engine.reset()
    assert engine.poll(7.01) is None and engine.pending is None


@pytest.mark.parametrize("delay", [0.05, 0.1, 0.25])
def test_real_async_and_replay_release_identical_supplied_completion_events(delay):
    class Worker(EmptyWorker):
        def infer(self, rgb):
            return dict(super().infer(rgb), latency_s=delay)

    replay, live = CueEngine(Worker(), replay=True), CueEngine(Worker(), replay=False)
    try:
        for engine in (replay, live):
            assert engine.submit(sensor(), 0)
        live.pending[1].result(timeout=1)
        for now in (4, 4 + delay - 0.001):
            assert replay.poll(now) is live.poll(now) is None
        a, b = replay.poll(4 + delay), live.poll(4 + delay)
        assert a[1:] == b[1:] and a[3] == 4 + delay
        replay.submit(sensor(7, 428), 1)
        live.submit(sensor(7, 428), 1)
        live.pending[1].result(timeout=1)
        replay.reset()
        live.reset()
        assert replay.poll(7 + delay) is live.poll(7 + delay) is None
    finally:
        replay.close()
        live.close()


def test_replay_live_event_equivalence_and_privileged_input_isolation():
    class RecordedEvents:
        def __init__(self):
            self.inputs = []
            self.generation = 0

        def reset(self):
            self.generation += 1

        def poll(self, now):
            return None

        def submit(self, sample, view):
            self.inputs.append(sample)
            return True

    first, second = (
        GeometryProvider(recipe(), RecordedEvents()),
        GeometryProvider(recipe(), RecordedEvents()),
    )
    for t, frame in ((4, 248), (4.016, 249), (7, 428)):
        clean = sensor(t, frame)
        poisoned = dict(
            clean,
            annotations={"hinge_origin": [999] * 3},
            handedness="oracle",
            asset_id="secret",
            phase=2,
            commands={"target": 999},
        )
        a, b = first.update(clean), second.update(poisoned)
        assert (a.valid, a.reason) == (b.valid, b.reason)
    assert all(set(v) == set(OBS_KEYS) for v in second.engine.inputs)
    assert not second.update(sensor(6, 429)).valid
    lost = sensor(8, 488)
    lost["valid_depth"][:] = False
    assert second.update(lost).reason == "missing_rgbd"
    assert second.encoding is None
    with pytest.raises(ValueError, match="qualified and frozen"):
        PerceptionBinding.from_dict(recipe().to_dict())


def test_observed_control_stops_on_loss_and_cannot_claim_force_feedback():
    from alexdoor_xas.qualification.synthetic_probe import ProbeSetup

    checks = ObservedControlChecks(ProbeSetup())
    sample = sensor(0, 8)
    limits = np.tile([-3, 3], (7, 1))
    tool = ObjectFrame(np.zeros(3), np.eye(3))
    estimate = complete_state()
    assert checks.check(sample, None, tool, limits, "inspect", np.zeros(2)) == ""
    assert (
        checks.check(sample, replace(estimate, valid=False), tool, limits, "push", np.zeros(2))
        == "invalid_perception"
    )
    assert (
        checks.check(sample, complete_state(0), tool, limits, "contact", np.zeros(2))
        == "force_feedback_unavailable"
    )
    sample["joint_position"][0] = 4
    assert (
        checks.check(sample, estimate, tool, limits, "approach", np.zeros(2))
        == "joint_limit_violation"
    )


@pytest.mark.parametrize("bottom_frame", [False, True])
def test_dense_edges_preserve_leaf_bottom_without_absorbing_fixed_frame(bottom_frame):
    from alexdoor_xas.perception.geometry import fuse_surface

    sample = sensor()
    h, w = 400, 200
    sample["rgb"] = np.ones((h, w, 3), np.uint8)
    sample["depth_m"] = np.full((h, w, 1), 2.0)
    sample["valid_depth"] = np.ones((h, w, 1), bool)
    sample["intrinsics"] = np.array([[100, 0, 100], [0, 100, 200], [0, 0, 1]])
    sample["camera_world"] = np.array(
        [[0, 0, 1, 0], [1, 0, 0, 0], [0, -1, 0, 1], [0, 0, 0, 1]], float
    )
    sample["depth_m"][90:296, 55:146] = 1.0
    # The segmentation includes fixed profiles. Their depth is distinct from the leaf.
    sample["depth_m"][80:90, 50:151] = 0.94
    if bottom_frame:
        sample["depth_m"][296:315, 50:151] = 0.94
    mask = np.zeros((h, w), bool)
    mask[80:315, 50:151] = True
    yy, xx = np.mgrid[120:250:3, 60:140:3]
    points = deproject(
        np.ones(xx.size),
        np.c_[xx.ravel(), yy.ravel()],
        sample["intrinsics"],
        sample["camera_world"],
    )
    edges, edge_points = extent_edges(
        points, np.array([1.0, 0, 0]), mask, sample, 0.004, return_points=True
    )
    assert edges["height_0"] == pytest.approx(0.05, abs=0.004)
    assert edges["height_1"] == pytest.approx(2.10, abs=0.004)
    first = Surface(
        points,
        np.array([1.0, 0, 0]),
        1.0,
        np.ones(8) / np.sqrt(8),
        points[:4],
        np.eye(8)[:4],
        0.001,
        1.0,
        {0},
        edges,
        edge_points,
    )
    second = replace(
        first,
        points=points + [0, 0, 0.1],
        anchors=points[-4:],
        features=np.eye(8)[4:],
        views={1},
        edge_points={},
    )
    fused = fuse_surface(first, second, recipe().config, 1)
    assert len(fused.anchors) == len(fused.features) == 8
    np.testing.assert_allclose(fused.anchors[-4:], second.anchors)
    assert fused.bounds[0, 2] == pytest.approx(0.05, abs=0.004)
    assert dimensions_supported(fused, 0.01)
    # The floor intersects the leaf plane in a thin line inside a leaked mask.
    floor_rows = np.arange(296, h)
    sample["depth_m"][296:, :, 0] = (100 / (floor_rows - 200))[:, None]
    mask[296:] = True
    _, _, dense = extent_edges(
        points, np.array([1.0, 0, 0]), mask, sample, 0.004, return_support=True
    )
    assert dense[:, 2].min() >= 0.045
    # A clipped leaf containing an internal recess must not acquire a false bottom edge.
    sample["depth_m"][296:, 55:146] = 1.0
    sample["depth_m"][230:240, 60:140] = 1.02
    mask[296:, 55:146] = True
    assert "height_0" not in extent_edges(points, np.array([1.0, 0, 0]), mask, sample, 0.004)


def test_scan_semantics_starts_at_first_observation_and_covers_between_hold_views():
    engine = CueEngine(EmptyWorker(), replay=True)
    provider = GeometryProvider(recipe(), engine)
    provider.update(sensor(0, 8))
    assert engine.pending[0][1]["time_s"] == 0
    provider.update(sensor(0.1, 14))
    assert engine.pending is None
    provider.update(sensor(0.2, 20))
    assert engine.pending[0][1]["time_s"] == 0.2
    provider.update(sensor(25.1, 1514))
    assert engine.pending is None
    assert not provider.last_estimate.valid


def test_production_surface_extraction_keeps_dense_extents_and_rejects_parallel_frame_fusion():
    from alexdoor_xas.perception.geometry import similar_surface, surfaces

    sample = sensor()
    sample["camera_world"][:3, :3] = [[0, 0, 1], [1, 0, 0], [0, -1, 0]]
    cue = EmptyWorker().infer(sample["rgb"])
    cue["masks"] = [np.packbits(np.ones((16, 16), bool)).tobytes()]
    cue["tokens"] = (np.ones((196, 384), np.float32) / np.sqrt(384)).astype(np.float32).tobytes()
    config = dict(recipe().config, min_points=12)
    parts = surfaces(cue, sample, config)
    assert parts and len(parts[0].extent_points)
    first = parts[0]
    frame = replace(first, points=first.points + [0.02, 0, 0], offset=first.offset + 0.02)
    assert not similar_surface(first, frame, config)
    # Same registered geometry can associate despite different appearance descriptors.
    same = replace(first, descriptor=-first.descriptor)
    assert similar_surface(first, same, config)


def test_scoring_retains_finite_rejected_errors_and_maximum():
    errors = np.zeros((4, len(COMPONENTS)))
    errors[0, 0] = 0.1
    accepted = np.array([False, True, True, True])
    result = score(errors, accepted, np.full(4, "observed"), np.ones(4, bool))
    assert result["errors_rejected"]["hinge_origin_m"]["maximum"] == 0.1
    assert result["errors_all_finite"]["hinge_origin_m"]["n"] == 4
    assert result["errors_accepted"]["hinge_origin_m"]["maximum"] == 0
    assert result["valid_coverage"] == 0.75 and not result["offline_passed"]
    assert quantiles(np.array([np.nan, 0.05, 0.1]))["maximum"] == 0.1


def test_static_provider_never_publishes_policy_state_or_requests_cues_after_scan():
    engine = CueEngine(EmptyWorker(), replay=True)
    provider = GeometryProvider(recipe(), engine)
    estimate = provider.update(sensor(25, 1508))
    assert not estimate.valid and estimate.reason == "static_scan_only"
    provider.consume_results(25.05)
    calls = provider.diagnostics["semantic_calls"]
    estimate = provider.update(sensor(25.1, 1514))
    assert engine.pending is None and provider.encoding is None
    assert provider.diagnostics["semantic_calls"] == calls
    assert estimate.operational is estimate.local is estimate.frame is None
