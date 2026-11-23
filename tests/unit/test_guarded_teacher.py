"""Guard rejection must precede dispatch; measured previews define admission."""

from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch

from alexdoor_xas.recording.guarded_teacher import GuardedTeacherRecorder


def recorder(monkeypatch, *, admitted=True, safety=None):
    import alexdoor_xas.policies.observations as observations

    monkeypatch.setattr(observations, "require_features", lambda *args: None)
    r = GuardedTeacherRecorder.__new__(GuardedTeacherRecorder)
    r.observer = SimpleNamespace(binding=object())
    r.latest_observation = object()
    r.admitted_commands, r.guard_refusal = 0, None
    r.brake_anchor = None
    r.tracker_stream = None
    q = torch.zeros((1, 9))
    target = torch.full((1, 7), 0.001)
    r.env = SimpleNamespace(
        device="cpu",
        capture=SimpleNamespace(sample=SimpleNamespace(joint_position=q)),
        preview_targets=Mock(return_value=(target, None, None)),
    )
    r.io = SimpleNamespace(
        safety_reason=Mock(return_value=safety),
        tool_pose=Mock(),
        execute=Mock(),
        response_violation=None,
    )
    r.motion = Mock(return_value=SimpleNamespace(admitted=admitted, reason="unsupported_sweep"))
    return r


def test_admitted_preview_executes_the_requested_pose(monkeypatch):
    r = recorder(monkeypatch)
    r.execute(np.zeros(3), np.eye(3), "approach")
    np.testing.assert_allclose(r.motion.call_args.args[2], np.full(7, 0.001))
    command = r.io.execute.call_args.args[0]
    assert command.kind == "pose" and command.admission.admitted
    assert r.admitted_commands == 1


@pytest.mark.parametrize(
    "safety,admitted", [("outside_verified_robot_stop_load_domain", True), (None, False)]
)
def test_refusal_never_dispatches(monkeypatch, safety, admitted):
    r = recorder(monkeypatch, safety=safety, admitted=admitted)
    with pytest.raises(ValueError):
        r.execute(np.zeros(3), np.eye(3), "contact")
    r.io.execute.assert_not_called()
    assert r.admitted_commands == 0 and r.guard_refusal


def test_brake_keeps_the_first_measured_joint_anchor(monkeypatch):
    r = recorder(monkeypatch)
    r.env.preview_targets.return_value = (torch.zeros((1, 7)), None, None)
    r.execute(np.zeros(3), np.eye(3), "approach", joint_brake=True)
    r.env.capture.sample.joint_position[0, :7] = 0.0001
    r.execute(np.zeros(3), np.eye(3), "approach", joint_brake=True)
    command = r.io.execute.call_args.args[0]
    assert command.kind == "joints"
    np.testing.assert_allclose(command.value, np.full(7, -0.0001))
    np.testing.assert_allclose(r.brake_anchor, 0)


def test_live_observation_update_preserves_native_capture_method(monkeypatch):
    from alexdoor_xas.recording.teacher import TemporalExpertRecorder

    r = recorder(monkeypatch)
    monkeypatch.setattr(TemporalExpertRecorder, "observation", lambda self, env: "raw pixels")
    monkeypatch.setattr(TemporalExpertRecorder, "_state", lambda self: self.observation(self.env))
    r.phase = "approach"
    r.io.observe = Mock(return_value="actual sensor")
    r.observer.update = Mock(return_value="policy observation")
    r._state()
    r._state()
    assert callable(r.observation)
    assert r.latest_observation == "policy observation"
    r.observer.update.assert_called_with("actual sensor")


def test_recording_detaches_without_waiting_for_operational_models(tmp_path):
    r = GuardedTeacherRecorder.__new__(GuardedTeacherRecorder)
    r.env = SimpleNamespace(event_sink=object())
    r.writer = Mock()
    geometry = Mock()
    r.operational = (None, None, geometry, None, None, None)
    r.close_recording()
    assert r.env.event_sink is None
    r.writer.close.assert_called_once()
    geometry.close.assert_not_called()
    assert r.operational is not None  # Model teardown remains a later operation.
