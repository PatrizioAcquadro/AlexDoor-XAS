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


def test_interruption_is_raised_after_returning_from_Kit_callback(monkeypatch):
    import runpy
    import signal

    import pytest

    api = runpy.run_path("scripts/pilot_physics.py")
    handlers, returned = {}, []
    monkeypatch.setattr(signal, "signal", lambda name, handler: handlers.update({name: handler}))

    def original_step():
        handlers[signal.SIGTERM](signal.SIGTERM, None)
        returned.append(True)  # Kit returns normally; no exception can be swallowed inside it.
        return "stepped"

    env = SimpleNamespace(step=original_step)
    restore = api["install_interruption"](env)
    with pytest.raises(KeyboardInterrupt, match="signal"):
        env.step()
    assert returned == [True]
    with pytest.raises(KeyboardInterrupt):
        env.step()  # cancellation remains pending; do not advance another task step
    assert returned == [True]
    restore()
    assert env.step is original_step


def test_pending_interruption_allows_full_continuing_physics_stop(tmp_path, monkeypatch):
    import runpy
    import signal

    api = runpy.run_path("scripts/pilot_physics.py")
    handlers = {}
    monkeypatch.setattr(signal, "signal", lambda name, handler: handlers.update({name: handler}))
    env = NumericalStopIO()
    restore = api["install_interruption"](env)
    handlers[signal.SIGTERM](signal.SIGTERM, None)
    restore()  # the maintained CLI does this before its finally/stop path
    result = verify_stop_continuation(env, tmp_path / "interrupted-stop.npz")
    assert result["passed"] and env.tick == 4 and env.stops == 1
    with np.load(tmp_path / "interrupted-stop.npz") as saved:
        np.testing.assert_allclose(saved["native_time_s"], np.arange(1, 9) / 4)


def test_cli_interruption_exit_survives_Kit_interpreter_shutdown():
    import ast
    from pathlib import Path

    import pytest

    tree = ast.parse(Path("scripts/pilot_physics.py").read_text())
    footer = ast.Module(body=tree.body[-1].body, type_ignores=[])

    def interrupted():
        raise KeyboardInterrupt("requested")

    def exit_process(code):
        raise SystemExit(code)

    namespace = dict(
        main=interrupted,
        traceback=SimpleNamespace(print_exc=lambda: None),
        sys=SimpleNamespace(
            stdout=SimpleNamespace(flush=lambda: None), stderr=SimpleNamespace(flush=lambda: None)
        ),
        os=SimpleNamespace(_exit=exit_process),
    )
    with pytest.raises(SystemExit) as error:
        exec(compile(footer, "pilot_cli_exit", "exec"), namespace)
    assert error.value.code == 130


def test_initial_screen_preserves_refusal_before_stop(tmp_path, monkeypatch):
    import json
    import runpy

    import pytest

    from alexdoor_xas.qualification import clearance, stop, visibility
    from alexdoor_xas.recording import b1_runtime

    api = runpy.run_path("scripts/pilot_physics.py")
    env = NumericalStopIO()
    env.robot.data.soft_joint_pos_limits = env.limits
    env.contacts = SimpleNamespace(read=lambda: dict(forbidden=False, contacts=[]))
    env.capture = SimpleNamespace(sample=SimpleNamespace(rgb=np.zeros((1, 2, 2, 3), "u1")))
    monkeypatch.setattr(b1_runtime, "array", lambda x: np.asarray(x))
    monkeypatch.setattr(
        clearance, "DoorClearance", lambda env: SimpleNamespace(measure=lambda *a: 0.0)
    )
    seen = []
    monkeypatch.setattr(
        stop, "verify_stop_continuation", lambda *a, **kw: seen.append(1) or dict(passed=True)
    )
    support = dict(passed=True)
    monkeypatch.setattr(visibility, "measure_visibility", lambda *a: support)
    setup = SimpleNamespace(contact_fraction=0.295, contact_height=1.09, force_limit_n=80)
    for defect in ("soft_limit", "contact", "visibility", "none"):
        folder = tmp_path / defect
        folder.mkdir()
        env.robot.data.soft_joint_pos_limits = env.limits.clone()
        if defect == "soft_limit":
            env.robot.data.soft_joint_pos_limits[0, 0, 0] = 0.01
        env.contacts.read = lambda defect=defect: dict(
            forbidden=False, contacts=[{}] if defect == "contact" else []
        )
        support["passed"] = defect != "visibility"
        if defect == "none":
            api["screen_initial_state"](env, object(), setup, folder)
        else:
            with pytest.raises(ValueError, match="unsupported_initial_state"):
                api["screen_initial_state"](env, object(), setup, folder)
        assert json.loads((folder / "initial-state.json").read_text())["passed"] == (
            defect == "none"
        )
    assert seen == [1]  # Refused starts never advance a stop or teacher step.


def test_reset_joint_limits_allow_one_float32_step_but_reject_real_excursions():
    from alexdoor_xas.qualification.limits import within_float32_joint_limits

    limits = np.array([[-1.0, 1.0], [-2.0, 2.0]], dtype="f4")
    for bound, direction in ((0, -np.inf), (1, np.inf)):
        edge = limits[:, bound]
        one_step = np.nextafter(edge, np.float32(direction))
        two_steps = np.nextafter(one_step, np.float32(direction))
        assert within_float32_joint_limits(edge, limits)
        assert within_float32_joint_limits(one_step, limits)
        assert not within_float32_joint_limits(two_steps, limits)
        assert not within_float32_joint_limits(edge + np.sign(direction) * 0.001, limits)
    assert within_float32_joint_limits(np.zeros(2, "f4"), limits)
    assert not within_float32_joint_limits(np.array([np.nan, 0]), limits)
    assert not within_float32_joint_limits(np.zeros(2), limits[::-1, ::-1])


def test_initial_screen_applies_same_roundoff_rule_to_hard_and_soft_limits(tmp_path, monkeypatch):
    import json
    import runpy

    from alexdoor_xas.qualification import clearance, stop, visibility
    from alexdoor_xas.recording import b1_runtime

    api = runpy.run_path("scripts/pilot_physics.py")
    env = NumericalStopIO()
    env.robot.data.joint_pos[0, 0] = float(np.nextafter(np.float32(1), np.float32(np.inf)))
    env.robot.data.soft_joint_pos_limits = env.limits.clone()
    env.contacts = SimpleNamespace(read=lambda: dict(forbidden=False, contacts=[]))
    env.capture = SimpleNamespace(sample=SimpleNamespace(rgb=np.zeros((1, 2, 2, 3), "u1")))
    monkeypatch.setattr(b1_runtime, "array", lambda x: np.asarray(x))
    monkeypatch.setattr(
        clearance, "DoorClearance", lambda env: SimpleNamespace(measure=lambda *a: 0)
    )
    monkeypatch.setattr(stop, "verify_stop_continuation", lambda *a, **kw: dict(passed=True))
    monkeypatch.setattr(visibility, "measure_visibility", lambda *a: dict(passed=True))
    setup = SimpleNamespace(contact_fraction=0.295, contact_height=1.09, force_limit_n=80)
    api["screen_initial_state"](env, object(), setup, tmp_path)
    audit = json.loads((tmp_path / "initial-state.json").read_text())
    assert audit["passed"]
    assert audit["minimum_joint_margin_rad"] < 0
    assert audit["minimum_soft_joint_margin_rad"] == audit["minimum_joint_margin_rad"]


def test_collection_simulation_guard_counts_steps_across_local_clock_resets(monkeypatch):
    import runpy
    import signal

    import pytest

    api = runpy.run_path("scripts/pilot_physics.py")
    monkeypatch.setattr(signal, "signal", lambda *a: None)
    env = SimpleNamespace(step=lambda: None, step_dt=0.5, _time_s=0.0)
    restore = api["install_interruption"](env, max_sim_s=1.0)
    env.step()
    env._time_s = 0.0
    with pytest.raises(api["CollectionBudgetExceeded"], match="simulation_budget"):
        env.step()
    assert env.collection_physics_s == 1.0
    restore()
