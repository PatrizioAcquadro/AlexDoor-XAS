"""A loaded diagnostic must rebuild support and use only fresh observations."""

import json
from types import SimpleNamespace

import numpy as np
import pytest

from alexdoor_xas.qualification.synthetic_probe import ProbeSetup
from alexdoor_xas.qualification.temporal_loaded_start import loaded_phase_startup


def source(tmp_path):
    setup = ProbeSetup()
    parent = tmp_path / "qualified"
    (parent / "teacher").mkdir(parents=True)
    path = parent / "teacher/hold-state.npz"
    np.savez(
        path,
        stiffness=np.ones((7, 1)),
        damping=np.ones((7, 1)),
        metadata=json.dumps(
            dict(asset_id="door", split="train", episode_id="source", setup=setup.to_dict())
        ),
    )
    (parent / "report.json").write_text(json.dumps(dict(case="door", passed=True)))
    return path, setup


def test_loaded_start_requires_matching_qualified_selected_train_source(tmp_path):
    path, setup = source(tmp_path)
    with pytest.raises(ValueError, match="qualified matching"):
        loaded_phase_startup(None, setup, path, asset_id="other", output=tmp_path / "out")
    startup = loaded_phase_startup(None, setup, path, asset_id="door", output=tmp_path / "out")
    observer = SimpleNamespace(
        execution=SimpleNamespace(
            policy=SimpleNamespace(stats=SimpleNamespace(train_episode_ids=("different",)))
        )
    )
    with pytest.raises(ValueError, match="outside selected training"):
        startup(observer, None)


@pytest.mark.parametrize("force,passes", [(1.0, True), (0.0, False)])
def test_loaded_start_qualifies_after_reset_and_acquires_fresh_history(
    tmp_path, monkeypatch, force, passes
):
    path, setup = source(tmp_path)
    order = []
    env = SimpleNamespace(
        cfg=SimpleNamespace(prepared_door="door"),
        targets=np.full((1, 9), 0.001),
        arm_ids=list(range(7)),
        _motion_impedance=[(None, None, np.ones(1), np.ones(1))] * 7,
        capture=SimpleNamespace(sample=SimpleNamespace(joint_position=np.zeros((1, 9)))),
        contact_history=[
            dict(
                forbidden=False,
                contacts=[dict(category="positive", force_n=force, separation_m=0.0)],
            )
        ]
        * 4,
        load_samples=[SimpleNamespace(dq=np.zeros(7))] * 4,
    )
    clock = [10.0]

    def prepare(*args, **kwargs):
        assert kwargs == dict(qualify_start_stop=True)
        order.append("restore_and_stop_qualification")
        clock[0] = 0.0
        args[3].mkdir()

    monkeypatch.setattr("alexdoor_xas.qualification.synthetic_probe.prepare_probe", prepare)

    def source_step(env, setup, state):
        clock[0] += 1 / 60
        return SimpleNamespace(origin=np.zeros(3), rot=np.eye(3))

    monkeypatch.setattr(
        "alexdoor_xas.qualification.temporal_loaded_start._source_step", source_step
    )
    observed = [10.0]

    def reset():
        order.append("reset_history")
        observed.clear()

    def update(sensor):
        observed.append(sensor)
        return SimpleNamespace(time_s=sensor)

    observer = SimpleNamespace(
        execution=SimpleNamespace(
            policy=SimpleNamespace(stats=SimpleNamespace(train_episode_ids=("source",)))
        ),
        reset=reset,
        update=update,
    )
    io = SimpleNamespace(
        dt=1 / 60,
        safety=SimpleNamespace(reset=lambda: order.append("reset_monitor")),
        tool_pose=lambda: None,
        observe=lambda: clock[0],
        safety_reason=lambda _: "",
        hold_step=lambda: clock.__setitem__(0, clock[0] + 1 / 60),
    )
    initialization_commands = []
    io.step_recorder = lambda command, goal: initialization_commands.append(io.initializing_phase)
    startup = loaded_phase_startup(env, setup, path, asset_id="door", output=tmp_path / "out")
    observation, reason = startup(observer, io)
    assert len(initialization_commands) == round(setup.intermediate_preroll_s / io.dt)
    assert all(initialization_commands) and not io.initializing_phase
    np.testing.assert_array_equal(io.parked_joints, env.targets[0, :7])
    assert order == ["restore_and_stop_qualification", "reset_history", "reset_monitor"]
    assert observed[0] == 0 and observation.time_s == pytest.approx(setup.intermediate_preroll_s)
    assert len(observed) == round(setup.intermediate_preroll_s / io.dt) + 1
    assert bool(not reason) == passes
    report = json.loads((tmp_path / "out/fresh-loaded-preroll.json").read_text())
    assert report["old_history_imported"] is False and report["passed"] == passes
    assert not passes or report["scope"] == "initialization_only_not_learned_task_completion"


def test_loaded_state_cannot_replace_qualified_normal_gains(tmp_path):
    path, setup = source(tmp_path)
    observer = SimpleNamespace(
        execution=SimpleNamespace(
            policy=SimpleNamespace(stats=SimpleNamespace(train_episode_ids=("source",)))
        )
    )
    env = SimpleNamespace(_motion_impedance=[(None, None, np.ones(1) * 2, np.ones(1))] * 7)
    startup = loaded_phase_startup(env, setup, path, asset_id="door", output=tmp_path / "out")
    with pytest.raises(ValueError, match="qualified normal-motion gains"):
        startup(observer, None)


def test_loaded_observation_schedule_retains_all_gates_and_original_workspace_times():
    from alexdoor_xas.perception.provider import PrototypeRecipe
    from alexdoor_xas.policies.pilot_runtime import loaded_observation_recipe

    config = dict(
        scan_fusion="object-v1",
        inspection=dict(sample_times_s=[5, 10, 15, 20, 25]),
        physical_and_observation_limits=dict(gap=0.0001, position=0.01),
    )
    recipe = PrototypeRecipe(json.dumps(config))
    assert loaded_observation_recipe(recipe, None) is recipe
    loaded = loaded_observation_recipe(recipe, 1.0)
    np.testing.assert_allclose(
        loaded.config["inspection"]["sample_times_s"], [0, 0.2, 0.4, 0.6, 0.8]
    )
    assert (
        loaded.config["physical_and_observation_limits"]
        == config["physical_and_observation_limits"]
    )
    assert recipe.config == config
    with pytest.raises(ValueError, match="all inspection captures"):
        loaded_observation_recipe(recipe, 0.5)
