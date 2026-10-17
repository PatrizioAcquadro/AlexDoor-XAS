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
        self._time_s = 0.0
        self.targets = torch.zeros((1, 7))
        self.limits = torch.tensor([[[-1.0, 1.0]] * 7])
        self.robot = SimpleNamespace(
            joint_names=[f"joint_{i}" for i in range(7)],
            data=SimpleNamespace(joint_pos=torch.zeros((1, 7))),
        )
        self.load_samples = []
        self.contact_history = [dict(components="normal_only", forbidden=forbidden, contacts=[])]

    def stop_motion(self):
        self.stops += 1

    def tool_pose(self):
        return torch.tensor([[self.tick * 0.001, 0, 0]]), torch.tensor([[0.0, 0, 0, 1]])

    def step(self, command):
        assert self.stops == 1 and not command.any()
        self.tick += 1
        self._time_s += self.step_dt
        self.load_samples = [
            SimpleNamespace(
                q=np.zeros(7),
                dq=np.full(7, 0.001),
                ddq=np.zeros(7),
                expected_torque_nm=np.zeros(7),
                jacobian_world=np.zeros((6, 7)),
                feedback=SimpleNamespace(
                    timestamp_s=self._time_s - offset,
                    available_s=self._time_s - offset,
                    torque_nm=np.zeros(7),
                    healthy=True,
                ),
            )
            for offset in (self.step_dt / 2, 0)
        ]
        self.contact_history *= 2 if len(self.contact_history) == 1 else 1


def test_stop_advances_each_physics_interval_and_retains_separate_tail(tmp_path):
    env = NumericalStopIO()
    path = tmp_path / "stop.npz"
    result = verify_stop_continuation(env, path)
    assert result["passed"] and env.tick == 4 and env.stops == 1
    assert result["position_travel_max_m"] == np.float32(0.004)
    with np.load(path) as saved:
        assert saved["values"].shape == (4, 6)
        assert saved["joint_position"].shape == (4, 7)
        assert saved["native_joint_position"].shape == (8, 7)
        np.testing.assert_allclose(saved["native_time_s"], np.arange(1, 9) / 4)
        np.testing.assert_array_equal(saved["native_time_s"], saved["evaluator_time_s"])
        assert saved["anchor_time_s"] == 0.0


def test_a_contact_violation_survives_successful_settling(tmp_path):
    env = NumericalStopIO(forbidden=True)
    result = verify_stop_continuation(env, tmp_path / "stop.npz")
    assert not result["passed"] and "forbidden_contact" in result["violations"]
    assert env.tick == 4
