"""Continuation/latch bookkeeping only; physical stop qualification is separate."""

from types import SimpleNamespace

import numpy as np
import torch

from alexdoor_xas.qualification.stop import verify_stop_continuation


class NumericalStopIO:
    step_dt, device, arm_ids = 0.5, "cpu", list(range(7))
    cfg = SimpleNamespace(action_space=6)

    def __init__(self, *, forbidden=False):
        self.tick = self.stops = 0
        self.targets = torch.zeros((1, 7))
        self.limits = torch.tensor([[[-1.0, 1.0]] * 7])
        self.robot = SimpleNamespace(data=SimpleNamespace(joint_pos=torch.zeros((1, 7))))
        self.load_samples = [SimpleNamespace(dq=np.full(7, 0.001))] * 2
        self.contact_history = [dict(components="normal_only", forbidden=forbidden, contacts=[])]

    def stop_motion(self):
        self.stops += 1

    def tool_pose(self):
        return torch.tensor([[self.tick * 0.001, 0, 0]]), torch.tensor([[0.0, 0, 0, 1]])

    def step(self, command):
        assert self.stops == 1 and not command.any()
        self.tick += 1


def test_stop_advances_each_physics_interval_and_retains_separate_tail(tmp_path):
    env = NumericalStopIO()
    path = tmp_path / "stop.npz"
    result = verify_stop_continuation(env, path)
    assert result["passed"] and env.tick == 4 and env.stops == 1
    assert result["position_travel_max_m"] == np.float32(0.004)
    assert np.load(path)["values"].shape == (4, 6)


def test_a_contact_violation_survives_successful_settling(tmp_path):
    env = NumericalStopIO(forbidden=True)
    result = verify_stop_continuation(env, tmp_path / "stop.npz")
    assert not result["passed"] and "forbidden_contact" in result["violations"]
    assert env.tick == 4
