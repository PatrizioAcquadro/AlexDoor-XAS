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
    r.observation = object()
    r.admitted_commands, r.guard_refusal = 0, None
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
