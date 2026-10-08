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
