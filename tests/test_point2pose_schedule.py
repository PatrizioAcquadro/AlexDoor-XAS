"""Scheduling must preserve alternatives without publishing stale evidence."""

from types import SimpleNamespace

import numpy as np
import pytest

from alexdoor_xas.perception.point2pose_schedule import SelectedRegistration


def fixture():
    calls = []

    def extract(obj, *args, **kwargs):
        calls.append(obj.id)
        return "native"

    pipeline = SimpleNamespace(
        use_key_frame_graph=False,
        frontend=SimpleNamespace(
            frame_reg_mode="f2m", _extract_valid_key_points_mask_remove=extract
        ),
        objects=[SimpleNamespace(id=i, lost=False) for i in range(3)],
    )
    schedule = SelectedRegistration(pipeline)
    return pipeline, schedule, calls


def request(pipeline, index):
    return pipeline.frontend._extract_valid_key_points_mask_remove(
        pipeline.objects[index], *([None] * 7), frame_id=1
    )


def test_selected_every_frame_alternatives_deferred_and_audited():
    pipeline, schedule, calls = fixture()
    schedule.begin(1)
    assert request(pipeline, 0) == "native"
    state = np.random.get_state()
    deferred = request(pipeline, 1)
    assert deferred[0].size == 0 and deferred[-1]["registration_deferred"]
    np.testing.assert_array_equal(state[1], np.random.get_state()[1])
    assert calls == [0] and schedule.metadata()["evaluated_object_ids"] == [0]
    schedule.begin(9)
    assert request(pipeline, 1) == "native"
    assert schedule.reason == "periodic_audit"


def test_current_failure_expands_same_frame_and_next_frame_without_switching():
    pipeline, schedule, calls = fixture()
    schedule.begin(1)
    request(pipeline, 0)
    pipeline.objects[0].lost = True  # Native gates settle before object 1.
    assert request(pipeline, 1) == "native"
    assert request(pipeline, 2) == "native"
    assert schedule.reason == "selected_support_lost"
    schedule.begin(2)
    pipeline.objects[0].lost = False  # Recovery does not cancel an expanded frame.
    assert request(pipeline, 1) == "native"
    assert schedule.selected == 0 and calls == [0, 1, 2, 1]
    schedule.begin(3)
    assert request(pipeline, 1)[-1]["registration_deferred"]


def test_graph_or_frame_to_frame_cannot_silently_use_schedule():
    pipeline, _, _ = fixture()
    pipeline.use_key_frame_graph = True
    with pytest.raises(ValueError, match="graph_off_f2m"):
        SelectedRegistration(pipeline)
    pipeline.use_key_frame_graph = False
    pipeline.frontend.frame_reg_mode = "f2f"
    with pytest.raises(ValueError, match="graph_off_f2m"):
        SelectedRegistration(pipeline)
