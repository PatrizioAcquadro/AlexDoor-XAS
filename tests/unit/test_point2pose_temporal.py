"""Causal queue timing, full process reset and transported-zone failure semantics."""

from types import SimpleNamespace

import numpy as np
import pytest

from alexdoor_xas.perception.contracts import FieldSupport
from alexdoor_xas.perception.geometry import Surface, SurfaceObservation, deproject
from alexdoor_xas.perception.ipc import pack_array
from alexdoor_xas.perception.local_surface import select_planar_normal
from alexdoor_xas.perception.point2pose.runtime import Point2PoseEngine
from alexdoor_xas.perception.point2pose.tracking import PanelCandidate, PanelTracking
from alexdoor_xas.recording.b1 import OBS_KEYS


def sensor(t=0.0, frame=0):
    camera = np.eye(4)
    camera[:3, :3] = [[0, 0, 1], [-1, 0, 0], [0, -1, 0]]
    return dict(
        time_s=np.asarray(t),
        frame=np.asarray(frame),
        rgb=np.ones((80, 80, 3), np.uint8),
        depth_m=np.ones((80, 80, 1)),
        valid_depth=np.ones((80, 80, 1), bool),
        joint_position=np.zeros(9),
        joint_velocity=np.zeros(9),
        camera_world=camera,
        intrinsics=np.array([[60, 0, 40], [0, 60, 40], [0, 0, 1.0]]),
    )


class Worker:
    def __init__(self):
        self.closed = False
        self.requests = []

    def infer(self, captured, masks=None):
        self.requests.append((int(captured["frame"]), masks))
        return dict(latency_s=0.08, objects=[])

    def close(self):
        self.closed = True


def test_latest_queue_wait_is_included_and_episode_recreates_worker():
    workers = []

    def factory(_):
        workers.append(Worker())
        return workers[-1]

    engine = Point2PoseEngine(factory, replay=True)
    try:
        first = sensor()
        engine.submit(first, [np.ones((80, 80), bool)])
        first["rgb"][:] = 0
        assert not engine.submit(sensor(0.02, 1))
        assert not engine.submit(sensor(0.06, 2))
        event = engine.poll(0.1)
        assert event[0]["rgb"].all()
        assert event[0]["rgb"].flags.writeable is False
        assert engine.poll(0.15) is None
        ready = engine.pending[1]["available_s"]
        event = engine.poll(ready)
        assert int(event[0]["frame"]) == 2
        assert event[3] == pytest.approx(0.18 + event[2]["snapshot_copy_s"])
        engine.reset()
        assert workers[0].closed
        assert engine.pending is engine.latest is engine.worker is None
        assert not engine.submit(sensor(0.2, 3))
        engine.submit(sensor(0.2, 3), [np.ones((80, 80), bool)])
        assert len(workers) == 2
    finally:
        engine.close()


def make_surface(captured):
    mask = np.zeros((80, 80), bool)
    mask[15:65, 15:65] = True
    v, u = np.nonzero(mask)
    points = deproject(
        np.ones(len(u)), np.c_[u, v], captured["intrinsics"], captured["camera_world"]
    )
    observation = SurfaceObservation(
        int(captured["frame"]),
        float(captured["time_s"]),
        float(captured["time_s"]),
        0,
        mask.shape,
        np.packbits(mask).tobytes(),
        captured["intrinsics"].copy(),
        captured["camera_world"].copy(),
        points[::20],
        {},
    )
    return Surface(
        points,
        np.array([1.0, 0, 0]),
        1.0,
        0.0,
        1.0,
        observations=(observation,),
    )


def test_normal_refit_uses_dominant_distributed_core_on_tracked_material():
    y, z = np.meshgrid(np.linspace(-0.12, 0.12, 18), np.linspace(-0.18, 0.18, 25))
    face = np.c_[np.zeros(y.size), y.ravel(), z.ravel()]
    bevel = face.copy()
    bevel[:, 0] = 0.004 + 0.12 * (bevel[:, 1] + 0.12)
    tilted_reference = np.array([np.cos(0.03), np.sin(0.03), 0.0])
    normal, radius = select_planar_normal(
        np.r_[face, bevel[::3]], 14, 120, 0.001, tilted_reference, 0.055
    )
    np.testing.assert_allclose(normal, [1, 0, 0], atol=1e-8)
    assert radius < 0.011
    _, ambiguous_radius = select_planar_normal(
        np.r_[face, face + [0.008, 0, 0]], 14, 120, 0.001, tilted_reference, 0.055
    )
    assert ambiguous_radius == 0.055


def test_normal_alignment_respects_publication_age_and_rejects_disjoint_frames():
    tracker = PanelTracking(
        SimpleNamespace(reset=lambda: None),
        dict(local_surface_constraint=dict(point_error_m=0.001, normal_model_bound_rad=0.001)),
        calibration_bound=(0.0, 0.002),
        relative_speed_bound=(0.1, 0.15),
    )
    first = ((1.0, 0.0, 0.0), np.deg2rad(2), FieldSupport(1.0, 1.0, 1.0, 0))
    other = ((0.0, 1.0, 0.0), np.deg2rad(2), FieldSupport(1.03, 1.04, 1.03, 0))
    candidate = SimpleNamespace(normal_observations=(first, other))
    tracker.candidates = [candidate]
    tracker.publish_normal_support(1.03, 1.06)
    assert tracker.normal_alignment(candidate, 1.05).cones == (
        (first[0], pytest.approx(first[1] + 0.003 + 0.15 * 0.05)),
    )
    with pytest.raises(ValueError, match="inconsistent_local_normal_observations"):
        tracker.normal_alignment(candidate, 1.06)
    assert tracker.normal_alignment(candidate, 1.20) is None


def test_local_tracker_publishes_normal_evidence_with_current_depth():
    tracker = PanelTracking(
        SimpleNamespace(reset=lambda: None),
        dict(
            local_surface_constraint=dict(point_error_m=0.001, normal_model_bound_rad=0.001),
            min_points=120,
            plane_tolerance_m=0.004,
        ),
        calibration_bound=(0.0, 0.001),
        relative_speed_bound=(0.1, 0.15),
    )
    first = sensor(1, 1)
    root = make_surface(first)
    root.edge_points["width_0"] = root.points[::50]
    root.observations[0].semantic_id = 0
    initial = np.eye(4)
    initial[0, 3] = 1.0
    candidate = PanelCandidate(
        "panel",
        root,
        initial,
        np.linalg.inv(first["camera_world"]) @ initial,
        np.eye(4),
        FieldSupport(1.0, 1.0, 1.0, 0, 0.004, 0.001),
        pose=initial,
        camera_world=first["camera_world"],
        state="tracked",
        ownership="observed_rigid_material",
        sam3_id=0,
    )
    tracker.candidates = [candidate]
    tracker.select_zone("panel", 1.0, source="policy")
    source = root.points[::100] @ first["camera_world"][:3, :3]
    measured = np.ones(len(source), bool)
    obj = dict(
        source_points=source,
        current_points=source,
        inliers=measured,
        measured_correspondences=measured,
    )
    for frame, timestamp in enumerate((1.0, 1.03), 1):
        captured = sensor(timestamp, frame)
        surface = make_surface(captured)
        surface.edge_points["width_0"] = surface.points[::50]
        observation = surface.observations[0]
        observation.semantic_id = 0
        candidate.support = FieldSupport(timestamp, timestamp, timestamp, 0)
        tracker.constrain_local(
            captured,
            dict(
                objects=[obj],
                local_surfaces=[surface],
                cue=dict(ids=[0], masks=[observation.support_mask], shape=observation.shape),
            ),
        )
        tracker.publish_normal_support(timestamp, timestamp + 0.01)
    patch = tracker.local_state(1.05).patches[0]
    assert patch.tracking_state == "tracking"
    assert len(patch.normal_alignment.cones) == 3
    assert patch.normal_alignment.evaluated_s == 1.05
    assert patch.normal_alignment.cones[-1][1] < np.deg2rad(1)
    assert candidate.normal_observations[0][2].available_s == 1.01
    tracker.select_zone("panel", 1.06, source="policy")
    assert tracker.local_state(1.06).patches[0].normal_alignment is None


def test_zone_without_local_texture_and_loss_does_not_refresh_support():
    engine = Point2PoseEngine(lambda _: Worker(), replay=True)
    config = dict(plane_tolerance_m=0.004, voxel_m=0.004, association_angle_deg=5)
    tracking = PanelTracking(engine, config)
    tracking.reset(7)
    captured = sensor()
    surface = make_surface(captured)
    try:
        assert tracking.initialize([surface], captured)
        candidate = tracking.candidates[0]
        original = candidate.initial_world.copy()
        surface.points[:] = 99
        assert not np.all(candidate.surface.points == 99)
        source = np.array(
            [
                [-0.25, -0.25, 1],
                [0.25, -0.25, 1],
                [0.25, 0.25, 1],
                [-0.25, 0.25, 1],
                [0, 0.2, 1],
                [0.2, 0, 1],
            ]
        )
        obj = dict(
            object_id=0,
            camera_from_map=pack_array(np.eye(4)),
            lost=False,
            source_points=pack_array(source),
            current_points=pack_array(source),
            measured_correspondences=pack_array(np.ones(6, bool)),
            inliers=pack_array(np.ones(6, bool)),
        )
        result = dict(objects=[obj], latency_s=0.05, capture_s=0.0)
        tracking.consume(captured, result, 0.05)
        state = tracking.local_state(0.1)
        assert state.patches[0].world_pose is not None
        assert state.patches[0].pose_support.position_bound_m is None  # missing FK bound
        assert state.patches[0].identity_support.reason == "ambiguous_leaf_ownership"
        np.testing.assert_allclose(candidate.pose, original, atol=1e-12)
        articulation = tracking.diagnostics["observed_articulation"][candidate.candidate_id]
        assert articulation["static_reference"]["support"]["supported_s"] == 0.0
        assert tracking.local_state(0.151).patches[0].world_pose is None
        tracking.consume(sensor(0.1, 1), dict(result, capture_s=0.1), 0.15)
        assert len(tracking.motion_samples[candidate.candidate_id]) == 1
        articulation = tracking.diagnostics["observed_articulation"][candidate.candidate_id]
        assert articulation["angle"]["supported_s"] == 0.1
        assert articulation["static_reference"]["support"]["supported_s"] == 0.0
        obj["lost"] = True
        tracking.consume(sensor(0.2, 1), dict(result, capture_s=0.2), 0.25)
        assert candidate.support.supported_s == 0.1
        assert articulation["angle"] is None
        assert tracking.local_state(0.25).patches[0].world_pose is None
        tracking.reset(8)
        assert not tracking.candidates and tracking.selection is None
        assert tracking.generation == 8
    finally:
        engine.close()


def test_mismatched_mask_acquisition_is_rejected_before_tracking():
    captured = sensor()
    surface = make_surface(captured)
    surface.observations[0].frame = 9
    engine = Point2PoseEngine(lambda _: Worker(), replay=True)
    tracking = PanelTracking(
        engine, dict(plane_tolerance_m=0.004, voxel_m=0.004, association_angle_deg=5)
    )
    try:
        with pytest.raises(ValueError, match="unsynchronized"):
            tracking.initialize([surface], captured)
    finally:
        engine.close()


def test_support_qualification_does_not_become_slide_tolerance():
    support = FieldSupport(0, 0, 0, 0, 0.0101, 0)
    with pytest.raises(ValueError, match="excessive"):
        support.require_bounds(position=True, qualification=True)
    assert set(sensor()) == set(OBS_KEYS)


def test_initialization_completion_uses_release_tick_not_old_cue_time():
    engine = Point2PoseEngine(lambda _: Worker(), replay=True)
    tracking = PanelTracking(
        engine, dict(plane_tolerance_m=0.004, voxel_m=0.004, association_angle_deg=5)
    )
    captured = sensor()
    try:
        tracking.initialize([make_surface(captured)], captured, start_s=0.12)
        assert engine.poll(0.199) is None
        ready = engine.pending[1]["available_s"]
        assert ready >= 0.2
        assert engine.poll(ready)[3] == ready
    finally:
        engine.close()


def test_redundant_observations_are_not_independent_candidates():
    from copy import deepcopy

    engine = Point2PoseEngine(lambda _: Worker(), replay=True)
    tracking = PanelTracking(
        engine,
        dict(
            plane_tolerance_m=0.004,
            voxel_m=0.004,
            association_distance_m=0.025,
            association_angle_deg=5,
            min_points=120,
        ),
    )
    captured = sensor()
    surface = make_surface(captured)
    try:
        tracking.initialize([surface, deepcopy(surface)], captured)
        assert len(tracking.candidates) == 1
        assert tracking.diagnostics["redundant_observations"] == 1
    finally:
        engine.close()


def test_worker_ipc_excludes_annotations_commands_and_asset_identity(monkeypatch):
    from types import SimpleNamespace

    import alexdoor_xas.perception.point2pose.runtime as runtime

    worker = object.__new__(runtime.Point2PoseWorker)
    worker.process = SimpleNamespace(stdin=object(), stdout=object())
    worker.first_request = False
    worker.gpu_resident_bytes = worker.gpu_sampled_peak_bytes = None
    requests = []
    monkeypatch.setattr(runtime, "send", lambda _, value: requests.append(value))
    monkeypatch.setattr(runtime, "receive", lambda _: dict(latency_s=0.02, objects=[]))
    captured = dict(
        sensor(),
        annotations={"hinge": [999, 999, 999]},
        command=np.ones(6),
        asset_id="must-never-cross",
        expected_pose=np.eye(4),
    )
    worker.infer(captured, [np.ones((80, 80), bool)])
    assert set(requests[0]["sensor"]) == set(OBS_KEYS)
    assert not set(captured).difference(OBS_KEYS).intersection(requests[0])
    assert requests[0]["mask_frame"] == 0 and requests[0]["mask_time_s"] == 0


def test_thin_uninitializable_hypothesis_does_not_discard_other_candidates():
    from copy import deepcopy

    captured = sensor()
    good = make_surface(captured)
    thin = deepcopy(good)
    mask = np.zeros((80, 80), bool)
    mask[15:65, 39:41] = True
    thin.observations[0].support_mask = np.packbits(mask).tobytes()
    engine = Point2PoseEngine(lambda _: Worker(), replay=True)
    tracker = PanelTracking(
        engine,
        dict(
            plane_tolerance_m=0.004,
            voxel_m=0.004,
            association_angle_deg=5,
            association_distance_m=0.025,
            min_points=120,
        ),
    )
    try:
        tracker.initialize([thin, good], captured)
        assert len(tracker.candidates) == 1
        assert (
            tracker.diagnostics["initialization_rejections"][0]["reason"]
            == "insufficient_candidate_core"
        )
        assert tracker.candidates[0].candidate_id == "candidate-1"
    finally:
        engine.close()


def test_prepared_models_are_recreated_before_new_episode_acquisitions():
    workers = []

    def factory(_):
        workers.append(Worker())
        return workers[-1]

    engine = Point2PoseEngine(factory, replay=True)
    try:
        engine.prepare()
        assert engine.worker is workers[0] and not engine.worker.first_request
        engine.reset()
        assert workers[0].closed and engine.worker is workers[1]
        assert not engine.worker.first_request
        assert engine.pending is engine.latest is None
    finally:
        engine.close()
    assert workers[-1].closed and len(workers) == 2  # close must not preload again


def test_new_arrival_replaces_queue_before_completed_request_dispatch(monkeypatch):
    worker = Worker()
    engine = Point2PoseEngine(lambda _: worker, replay=True)
    tracker = PanelTracking(
        engine, dict(plane_tolerance_m=0.004, voxel_m=0.004, association_angle_deg=5)
    )
    captured = sensor()
    try:
        tracker.initialize([make_surface(captured)], captured)
        monkeypatch.setattr(tracker, "consume", lambda *args: None)
        tracker.update(sensor(0.05, 1))
        tracker.update(sensor(0.10, 2))
        assert [frame for frame, _ in worker.requests] == [0, 2]
        assert engine.pending[0][1]["time_s"] == pytest.approx(0.10)
    finally:
        engine.close()


def test_sampled_support_keeps_losses_and_empty_availability_explicit():
    from alexdoor_xas.perception.diagnostics.replay import support_timeline

    rows = [
        dict(time_s=t, available=valid)
        for t, valid in [(0.0, True), (0.1, True), (0.2, False), (0.3, True)]
    ]
    support = support_timeline(rows)
    assert support["recoveries"] == 1
    assert support["longest_sampled_span_s"] == pytest.approx(0.1)
    assert [i["samples"] for i in support["intervals"]] == [2, 1]
    assert support_timeline([dict(time_s=0.0, available=False)])["intervals"] == []
