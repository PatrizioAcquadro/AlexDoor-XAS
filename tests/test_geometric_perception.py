"""Essential causal, metric and complete-state behavior without model execution."""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from alexdoor_xas.action.frames import ObjectFrame, rot_z
from alexdoor_xas.perception.contracts import DoorEstimate, validate_complete
from alexdoor_xas.perception.control import ObservedControlChecks
from alexdoor_xas.perception.evaluation import (
    COMPONENTS,
    LIMITS,
    campaign_summary,
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
    hinge_from_motion,
    plane_fit,
    project,
)
from alexdoor_xas.perception.provider import CueEngine, GeometryProvider, load_recipe
from alexdoor_xas.policies.common.b1_contract import PerceptionBinding
from alexdoor_xas.policies.observations import require_estimate
from alexdoor_xas.recording.b1 import OBS_KEYS


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


@pytest.mark.parametrize("sign", [-1, 1])
def test_motion_hinge_fit_both_hands_and_degenerate_motion(sign):
    hinge = np.array([0.025, sign * 0.45, 0.0])
    motions = []
    for degrees in (4, 7, 11):
        rotation = rot_z(sign * np.deg2rad(degrees))
        motions.append((rotation, hinge - rotation @ hinge, 0.0001))
    fitted, uncertainty = hinge_from_motion(motions, np.deg2rad(3), 0.0)
    np.testing.assert_allclose(fitted, hinge, atol=0.001)
    assert uncertainty < 0.01
    assert hinge_from_motion([(np.eye(3), np.zeros(3), 0)] * 4, np.deg2rad(3), 0) is None


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


def test_clipped_extent_is_not_a_measured_dimension_and_ambiguous_identity_rejects():
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
    provider = GeometryProvider(recipe(), CueEngine(EmptyWorker(), replay=True))
    provider.static = [s, replace(s, offset=1.04)]
    provider._select_panel()
    assert provider.closed is None and provider.panel is None
    assert provider.diagnostics["association_reason"] == "ambiguous_panel_jamb_wall"
    normal = np.array([1.0, 0.01, 0.1])
    np.testing.assert_allclose(contact_frame(normal)[:, 0], normal / np.linalg.norm(normal))


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


def test_partial_campaign_cannot_enable_dynamic_and_report_preserves_maximum(tmp_path):
    output = tmp_path / "door" / "nominal"
    output.mkdir(parents=True)
    errors = np.zeros((4, len(COMPONENTS)))
    accepted = np.ones(4, bool)
    reasons, causes = np.full(4, "observed"), np.full(4, "")
    phase = np.full(4, 2)  # contact
    np.savez(
        output / "predictions.npz",
        errors=errors,
        accepted=accepted,
        reasons=reasons,
        causes=causes,
        phase=phase,
    )
    metrics = score(errors, accepted, reasons, np.ones(4, bool))
    report = dict(
        asset_id="door",
        split="train",
        handedness="left",
        condition="nominal",
        observations=4,
        manipulation=metrics,
        phases={"contact": metrics},
        semantic_latency_s=quantiles(np.array([0.05, 0.1])),
        recovery_s=None,
        final_diagnostics={},
    )
    summary = campaign_summary([report], tmp_path, 50)
    assert not summary["complete"] and not summary["offline_passed"]
    assert summary["dynamic_status"] == "pending_complete_campaign"
    assert "0.100000" in (tmp_path / "report.md").read_text()


def sensor(t=4.0, frame=248):
    return dict(
        time_s=t,
        frame=frame,
        rgb=np.ones((16, 16, 3), np.uint8),
        depth_m=np.ones((16, 16, 1)),
        valid_depth=np.ones((16, 16, 1), bool),
        joint_position=np.zeros(9),
        joint_velocity=np.zeros(9),
        camera_world=np.eye(4),
        intrinsics=np.array([[20, 0, 8], [0, 20, 8], [0, 0, 1]]),
    )


class EmptyWorker:
    def infer(self, rgb):
        return dict(
            latency_s=0.05,
            shape=rgb.shape[:2],
            masks=[],
            tokens=bytes(196 * 384 * 4),
            token_shape=(196, 384),
            pixel_mapping=dict(scale=14, pad_x=0, pad_y=0),
        )


def recipe():
    root = Path(__file__).resolve().parents[1]
    return load_recipe(root / "configs/perception_geometry.json", root)


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

        def reset(self):
            pass

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
