"""One frontend, acquisition binding, ambiguity, static freeze and real reset."""

import threading

import numpy as np
import pytest
from test_point2pose_temporal import make_surface, sensor

from alexdoor_xas.perception import session as module
from alexdoor_xas.perception.ipc import pack_array
from alexdoor_xas.perception.point2pose.runtime import Point2PoseEngine
from alexdoor_xas.perception.session import PerceptionSession
from alexdoor_xas.recording.b1 import OBS_KEYS
from perception_helpers import recipe


class Sam:
    runtime = {}

    def __init__(self, identities):
        self.identities, self.calls, self.closed = iter(identities), [], False

    def infer(self, rgb, frame, time_s):
        self.calls.append((int(frame), float(time_s)))
        ids = next(self.identities)
        self.all_masks = np.ones((len(ids), *rgb.shape[:2]), bool)
        return dict(
            ids=ids,
            shape=rgb.shape[:2],
            latency_s=0.01,
            masks=[np.packbits(m).tobytes() for m in self.all_masks],
            sam3=dict(frame=int(frame), time_s=float(time_s)),
        )

    def close(self):
        self.closed = True


class Native:
    runtime = {}
    boot_latency_s = 0.0

    def __init__(self):
        self.calls, self.closed = [], False

    def infer(self, observed, masks):
        assert set(observed) == set(OBS_KEYS)
        self.calls.append((observed, masks))
        return dict(
            objects=[dict(object_id=0, lost=False)],
            latency_s=0.01,
            frame=int(observed["frame"]),
            capture_s=float(observed["time_s"]),
            masks=pack_array(np.stack(masks)),
        )

    def close(self):
        self.closed = True


@pytest.fixture
def geometry(monkeypatch):
    def extract(cue, observed, config):
        output = []
        for identity in cue["ids"]:
            surface = make_surface(observed)
            surface.observations[0].semantic_id = identity
            output.append(surface)
        return output

    monkeypatch.setattr(module, "surfaces", extract)


def test_search_selection_loss_and_return_share_one_frontend(tmp_path, geometry):
    sam, native = Sam([[], [4, 9], [4, 9], [4], [9]]), Native()
    session = PerceptionSession(
        "unused", recipe(), {}, tmp_path, sam_worker=sam, p2p_worker=native, initialize_after_s=4
    )
    try:
        assert session.infer(sensor(4, 0))["identity"]["state"] == "no_candidate"
        assert session.infer(sensor(4.1, 1))["identity"]["state"] == "selection_required"
        assert not native.calls
        session.select(9)
        seed = session.infer(sensor(4.2, 2))
        assert seed["binding"][0]["sam3_id"] == 9
        assert seed["binding"][0]["p2p_object_id"] == 0
        assert seed["initial_candidates"][0].geometry.acquired_s == 4.2
        assert (
            seed["initial_candidates"][0].static_reference.support
            == seed["initial_candidates"][0].geometry
        )
        np.testing.assert_array_equal(native.calls[0][1][0], np.ones((80, 80), bool))
        missing = session.infer(sensor(4.3, 3))
        assert missing["semantic_absent"]
        assert not native.calls[-1][1][0].any()
        returned = session.infer(sensor(4.4, 4))
        assert returned["identity"]["returned_same_id"]
        assert not returned["identity"]["material_recovery_verified"]
        assert len(sam.calls) == 5 and len(native.calls) == 3
        assert not sam.closed
        with pytest.raises(ValueError, match="requires_session_reset"):
            session.select(4)
    finally:
        session.close()
    assert sam.closed and native.closed


def test_inspection_freezes_scene_and_retains_surrounding_depth(tmp_path, geometry):
    session = PerceptionSession(
        "unused", recipe(), {}, tmp_path, sam_worker=Sam([[7], [7], [7]]), p2p_worker=Native()
    )
    try:
        initial = session.infer(sensor(4, 0))
        assert not initial["objects"]
        seed = session.infer(sensor(25, 1))
        assert seed["objects"]
        static_points = session.scan_memory.scene.copy()
        static_times = [float(s["time_s"]) for s in session.scan_memory.space.views.values()]
        moved = sensor(26, 2)
        moved["depth_m"][:] = 2
        session.infer(moved)
        np.testing.assert_array_equal(session.scan_memory.scene, static_points)
        assert [
            float(s["time_s"]) for s in session.scan_memory.space.views.values()
        ] == static_times
        np.testing.assert_array_equal(session.native.calls[-1][0]["depth_m"], moved["depth_m"])
    finally:
        session.close()


def test_insufficient_geometry_is_retryable_without_a_new_prompt(tmp_path, monkeypatch):
    sam = Sam([[3], [3]])
    calls = []

    def measured(cue, observed, config):
        calls.append(int(observed["frame"]))
        if len(calls) == 1:
            return []
        surface = make_surface(observed)
        surface.observations[0].semantic_id = 3
        return [surface]

    monkeypatch.setattr(module, "surfaces", measured)
    session = PerceptionSession(
        "unused", recipe(), {}, tmp_path, sam_worker=sam, p2p_worker=Native(), initialize_after_s=4
    )
    try:
        assert session.infer(sensor(4, 0))["identity"]["state"] == "insufficient_geometry"
        assert session.infer(sensor(4.1, 1))["initial_candidates"]
        assert sam.calls == [(0, 4), (1, 4.1)]
    finally:
        session.close()


def test_reset_during_inflight_kills_temporal_state_and_queues():
    workers = []

    class Blocking:
        def __init__(self):
            self.entered, self.stopped = threading.Event(), threading.Event()

        def infer(self, observed, masks):
            self.entered.set()
            self.stopped.wait(2)
            return dict(latency_s=0.01)

        def close(self):
            self.stopped.set()

    def factory(_):
        worker = Blocking()
        workers.append(worker)
        return worker

    engine = Point2PoseEngine(factory, replay=False, capture_session=True)
    try:
        engine.prepare()
        engine.submit(sensor(4, 0))
        assert workers[0].entered.wait(1)
        engine.submit(sensor(4.1, 1))
        engine.reset()
        assert workers[0].stopped.is_set()
        assert len(workers) == 2
        assert engine.pending is engine.latest is None
        assert engine.poll(100) is None
        assert engine.queue_stats["submitted"] == 0
    finally:
        engine.close()


def test_selected_id_survives_mask_order_changes(tmp_path, geometry):
    class OrderedSam(Sam):
        def infer(self, *args):
            cue = super().infer(*args)
            self.all_masks = np.stack([np.full((80, 80), i == 9, bool) for i in cue["ids"]])
            cue["masks"] = [np.packbits(m).tobytes() for m in self.all_masks]
            return cue

    native = Native()
    session = PerceptionSession(
        "unused", recipe(), {}, tmp_path, sam_worker=OrderedSam([[4, 9], [9, 4]]), p2p_worker=native
    )
    try:
        session.select(9)
        first = session.infer(sensor(25, 0))
        second = session.infer(sensor(26, 1))
        assert first["binding"][0]["sam3_id"] == second["identity"]["sam3_id"] == 9
        assert len(native.calls) == 2
        assert all(masks[0].all() for _, masks in native.calls)
    finally:
        session.close()


def test_static_snapshot_preserves_complete_publication_times(tmp_path, geometry):
    session = PerceptionSession(
        "unused", recipe(), {}, tmp_path, tracking=False, sam_worker=Sam([[7], [7]])
    )
    engine = Point2PoseEngine(lambda _: session, replay=True, capture_session=True)
    provider = module.SessionProvider(recipe(), engine, tracking=False)
    try:
        engine.prepare()
        publications = {}
        for index, capture in enumerate((4.0, 4.3)):
            provider.update(sensor(capture, index))
            assert provider.consume_results(engine.pending[1]["available_s"])
            publications[capture] = provider.diagnostics["available_time_s"]
            observations = [o for s in provider.static for o in s.observations]
            assert observations
            for observation in observations:
                assert observation.available_s == publications[observation.acquired_s]
        provider.reset(preload=False)
        assert provider.scan_available == {}
        assert not provider.static
    finally:
        provider.close()


def test_inspection_end_requires_explicit_unique_supported_reference(tmp_path, monkeypatch):
    from copy import deepcopy

    def alternatives(cue, observed, config):
        left = make_surface(observed)
        left.observations[0].semantic_id = 3
        mask = np.zeros((80, 80), bool)
        mask[15:65, 15:35] = True
        left.observations[0].support_mask = np.packbits(mask).tobytes()
        right = deepcopy(left)
        right.points = right.points + [0.1, 0, 0]
        mask[:] = False
        mask[15:65, 45:65] = True
        right.observations[0].support_mask = np.packbits(mask).tobytes()
        return [left, right]

    monkeypatch.setattr(module, "surfaces", alternatives)
    native, sam = Native(), Sam([[3]] * 4)
    session = PerceptionSession("unused", recipe(), {}, tmp_path, sam_worker=sam, p2p_worker=native)
    try:
        before = session.infer(sensor(24, 0))
        assert not before["initialization"]["inspection_complete"]
        assert len(before["geometric_choices"]) == 2
        ended = session.infer(sensor(25, 1))
        assert ended["initialization"]["inspection_complete"]
        assert ended["identity"]["state"] == "ambiguous_geometry"
        assert not ended["initialization"]["p2p_started"] and not native.calls
        session.select(3, reference_pixel=[0, 0])
        unsupported = session.infer(sensor(25.1, 2))
        assert unsupported["initialization"]["reference_state"] == "reference_unsupported"
        assert not native.calls
        session.select(3, reference_pixel=[20, 40])
        seed = session.infer(sensor(25.2, 3))
        assert seed["initialization"]["reference_state"] == "available"
        assert seed["initialization"]["p2p_started"] and len(native.calls) == 1
        assert len(sam.calls) == 4 and not sam.closed
        with pytest.raises(ValueError, match="requires_session_reset"):
            session.select(3, reference_pixel=[50, 40])
    finally:
        session.close()


def test_missing_static_selection_preserves_session_and_pending_identity(tmp_path, geometry):
    sam = Sam([[8], [7], [7]])
    session = PerceptionSession(
        "unused", recipe(), {}, tmp_path, sam_worker=sam, p2p_worker=Native()
    )
    try:
        session.select(7, "absent-reference")
        with pytest.raises(ValueError, match="requires_session_reset"):
            session.select(8)
        absent = session.infer(sensor(25, 0))
        assert absent["identity"]["state"] == "selected_identity_absent"
        for index in (1, 2):
            result = session.infer(sensor(25 + index / 10, index))
            assert result["identity"]["sam3_id"] == 7
            assert result["initialization"]["reference_state"] == "reference_unsupported"
            assert not result["initialization"]["p2p_started"]
        assert len(sam.calls) == 3 and not session.native.calls and not sam.closed
    finally:
        session.close()


def test_delayed_ingress_uses_publication_clock_without_refreshing_capture(tmp_path, geometry):
    session = PerceptionSession(
        "unused", recipe(), {}, tmp_path, tracking=False, sam_worker=Sam([[7], [7]])
    )
    engine = Point2PoseEngine(lambda _: session, replay=True, capture_session=True)
    provider = module.SessionProvider(recipe(), engine, tracking=False)
    try:
        engine.prepare()
        provider.update(sensor(4, 0), now=4.5)
        assert provider.consume_results(5)
        captured, result = provider.last_publication
        assert float(captured["time_s"]) == result["capture_s"] == 4
        assert result["queue_wait_s"] == 0
        assert result["ingress_delay_s"] == 0.5
        assert result["publication_poll_wait_s"] > 0
        assert result["publication_latency_s"] > 1
        first_support = provider.static[0].observations[0]
        first_publication = first_support.available_s
        provider.update(sensor(4.3, 1), now=5.3)
        assert provider.consume_results(6)
        assert provider.static[0].observations[0].available_s == first_publication
        assert provider.last_publication[1]["capture_s"] == 4.3
        provider.reset(preload=False)
        assert provider.last_publication is None
    finally:
        provider.close()


def test_synchronous_simulation_keeps_measured_clock_and_explicit_cadence(tmp_path, geometry):
    clock = [4.0]
    sam = Sam([[7], [7]])
    session = PerceptionSession("unused", recipe(), {}, tmp_path, tracking=False, sam_worker=sam)
    engine = Point2PoseEngine(
        lambda _: session, replay=False, capture_session=True, clock=lambda: clock[0]
    )
    provider = module.SessionProvider(
        recipe(), engine, tracking=False, synchronous_simulation=True, inference_period_s=0.1
    )
    try:
        engine.prepare()
        provider.update(sensor(4.0, 0))
        assert len(sam.calls) == 1
        support = provider.static[0].observations[0]
        assert support.acquired_s == support.available_s == 4.0
        assert provider.last_publication[1]["publication_clock_domain"] == "simulation"
        assert provider.last_publication[1]["latency_s"] > 0
        clock[0] = 4.05
        provider.update(sensor(4.05, 1))
        assert len(sam.calls) == 1
        clock[0] = 4.1
        provider.update(sensor(4.1, 2))
        assert len(sam.calls) == 2
        provider.reset(preload=False)
        assert provider.last_submitted_s is None
    finally:
        provider.close()


def test_synchronous_simulation_cannot_hide_a_free_running_clock(tmp_path, geometry):
    clock = [4.0]
    sam = Sam([[7]])
    infer = sam.infer

    def advancing(*args):
        result = infer(*args)
        clock[0] += 0.3
        return result

    sam.infer = advancing
    session = PerceptionSession("unused", recipe(), {}, tmp_path, tracking=False, sam_worker=sam)
    engine = Point2PoseEngine(
        lambda _: session, replay=False, capture_session=True, clock=lambda: clock[0]
    )
    provider = module.SessionProvider(recipe(), engine, tracking=False, synchronous_simulation=True)
    try:
        engine.prepare()
        with pytest.raises(ValueError, match="synchronous_simulation_clock_advanced"):
            provider.update(sensor(4.0, 0))
        assert provider.diagnostics["error"] == "synchronous_simulation_clock_advanced"
        assert not provider.static
    finally:
        provider.close()


def test_session_factory_forwards_robot_camera_calibration(tmp_path, monkeypatch):
    class Session:
        runtime = {}

        def __init__(self, *args, **kwargs):
            pass

        def close(self):
            pass

    monkeypatch.setattr(module, "PerceptionSession", Session)
    calibration = {"root_world": np.eye(4).tolist()}
    provider = module.session_provider(recipe(), "unused", calibration, tmp_path, replay=True)
    try:
        assert provider.tracking.calibration is calibration
        assert provider.tracking.calibration_bound is None
    finally:
        provider.close()


def test_observed_reference_selection_preserves_ties_overlap_and_mask_holes():
    from types import SimpleNamespace

    def surface(mask):
        return SimpleNamespace(observations=[SimpleNamespace(mask=lambda: mask)])

    large = np.ones((20, 30), bool)
    small = np.zeros_like(large)
    small[5:15, 10:20] = True
    pixel = module.observed_reference_pixel([surface(large), surface(small)])
    u, v = pixel
    assert large[v, u] and not small[v, u]
    assert 0 < u < 29 and 0 < v < 19
    assert module.observed_reference_pixel([surface(large), surface(large)]) is None
    # A larger union with no exclusive material cannot become a unique seed.
    left = large.copy()
    left[:, 16:] = False
    right = large.copy()
    right[:, :14] = False
    assert module.observed_reference_pixel([surface(large), surface(left), surface(right)]) is None


def test_provider_converts_source_pixel_centers_once_before_all_geometry(tmp_path, geometry):
    import json

    from alexdoor_xas.perception.provider import PrototypeRecipe

    config = recipe().config
    config["camera_pixel_center_offset_xy"] = [0.5, 0.5]
    binding = PrototypeRecipe(json.dumps(config))
    native = Native()
    session = PerceptionSession(
        "unused", binding, {}, tmp_path, sam_worker=Sam([[7]]), p2p_worker=native
    )
    engine = Point2PoseEngine(lambda _: session, replay=True, capture_session=True)
    provider = module.SessionProvider(binding, engine, tracking=False)
    observed = sensor(25, 0)
    raw = observed["intrinsics"].copy()
    try:
        provider.update(observed)
        assert provider.consume_results(engine.pending[1]["available_s"])
        captured, _ = provider.last_publication
        expected = raw.copy()
        expected[:2, 2] -= 0.5
        np.testing.assert_array_equal(captured["intrinsics"], expected)
        np.testing.assert_array_equal(observed["intrinsics"], raw)
        assert provider.static
        np.testing.assert_array_equal(provider.static[0].observations[0].intrinsics, expected)
    finally:
        provider.close()


def test_geometry_robot_exclusion_preserves_raw_depth_and_policy_validity(tmp_path, geometry):
    session = PerceptionSession(
        "unused", recipe(), {}, tmp_path, tracking=False, sam_worker=Sam([[7]])
    )
    engine = Point2PoseEngine(lambda _: session, replay=True, capture_session=True)
    removed = np.zeros((80, 80), bool)
    removed[20:30, 25:35] = True
    provider = module.SessionProvider(
        recipe(), engine, tracking=False, geometry_filter=lambda sensor, raw: removed
    )
    observed = sensor(4, 0)
    try:
        provider.update(observed)
        assert provider.consume_results(engine.pending[1]["available_s"])
        captured, _ = provider.last_publication
        assert not captured["valid_depth"][removed].any()
        np.testing.assert_array_equal(captured["depth_m"], observed["depth_m"])
        assert observed["valid_depth"].all()
    finally:
        provider.close()


def test_explicit_policy_protocol_selects_one_observed_zone_without_qualifying_ownership(
    tmp_path, geometry
):
    class MeasuredNative(Native):
        def infer(self, observed, masks):
            result = super().infer(observed, masks)
            points = np.array(
                [
                    [-0.2, -0.2, 1],
                    [0.2, -0.2, 1],
                    [0.2, 0.2, 1],
                    [-0.2, 0.2, 1],
                    [0, 0, 1],
                    [0.1, 0.1, 1],
                ]
            )
            result["objects"][0].update(
                camera_from_map=pack_array(np.eye(4)),
                source_points=pack_array(points),
                current_points=pack_array(points),
                measured_correspondences=pack_array(np.ones(6, bool)),
                inliers=pack_array(np.ones(6, bool)),
            )
            return result

    session = PerceptionSession(
        "unused", recipe(), {}, tmp_path, sam_worker=Sam([[7]]), p2p_worker=MeasuredNative()
    )
    engine = Point2PoseEngine(lambda _: session, replay=True, capture_session=True)
    provider = module.SessionProvider(recipe(), engine, local_selection_source="policy")
    try:
        provider.update(sensor(25, 0))
        ready = engine.pending[1]["available_s"]
        assert provider.consume_results(ready)
        selection = provider.tracking.selection
        assert selection.source == "policy" and selection.selected_s == ready
        assert selection.patch_id == provider.tracking.candidates[0].candidate_id
        assert provider.tracking.candidates[0].ownership == "ambiguous_leaf_candidate"
        provider.reset(preload=False)
        assert provider.tracking.selection is None
    finally:
        provider.close()
