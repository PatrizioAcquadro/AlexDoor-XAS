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


@pytest.mark.parametrize(
    "force,load_ready_s,passes",
    [(1.0, None, True), (0.0, None, False), (1.0, 0.9, True), (1.0, 2.0, False)],
)
def test_loaded_start_qualifies_after_reset_and_acquires_fresh_history(
    tmp_path, monkeypatch, force, load_ready_s, passes
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
        return SimpleNamespace(
            time_s=sensor,
            estimate=SimpleNamespace(
                local=SimpleNamespace(selection="selected") if sensor >= 0.8 else None
            ),
        )

    observer = SimpleNamespace(
        execution=SimpleNamespace(
            policy=SimpleNamespace(
                stats=SimpleNamespace(train_episode_ids=("fit-only",)),
                training_episode_ids=("fit-only", "source"),
            )
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
    load_calls = []
    if load_ready_s is not None:

        def measured_load(io, feedback):
            load_calls.append(clock[0])
            if clock[0] < load_ready_s - 1e-9:
                raise ValueError("unavailable_native_material_identity")
            return "supported_load"

        io.load_source = measured_load
        io.robot_feedback = lambda: None
        io.safety.checks = SimpleNamespace(
            contact_monitor=SimpleNamespace(contact_load=lambda measured, time_s: None)
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
    if load_ready_s is not None:
        assert load_calls[0] < 0.9 and load_calls[-1] == pytest.approx(1.0)
        assert any(row["operational_load_reason"] for row in report["rows"])
        assert report["rows"][-1]["operational_load_ready"] == passes
        assert passes or "observed_load_unavailable" in reason


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


def test_loaded_observation_schedule_retains_all_gates_and_original_workspace_times(b1_binding):
    from alexdoor_xas.perception.provider import PrototypeRecipe
    from alexdoor_xas.policies.pilot_runtime import loaded_observation_recipe

    config = dict(
        scan_fusion="object-v1",
        inspection=b1_binding.config["inspection"],
        physical_and_observation_limits=dict(gap=0.0001, position=0.01),
    )
    recipe = PrototypeRecipe(json.dumps(config))
    assert loaded_observation_recipe(recipe, None) is recipe
    loaded = loaded_observation_recipe(recipe, 1.0)
    times = np.asarray(loaded.config["inspection"]["sample_times_s"])
    assert len(times) == len(recipe.config["inspection"]["sample_times_s"])
    np.testing.assert_allclose(times[[0, -1]], [0.1, 0.8])
    assert np.all(np.diff(times) >= 0.1 - 1e-9)
    assert (
        loaded.config["physical_and_observation_limits"]
        == config["physical_and_observation_limits"]
    )
    assert recipe.config == config
    with pytest.raises(ValueError, match="all inspection captures"):
        loaded_observation_recipe(recipe, 0.5)


def test_loaded_acquisition_uses_a_real_matching_observer_and_io_contract(b1_binding):
    from alexdoor_xas.perception.provider import PrototypeRecipe
    from alexdoor_xas.policies.encoding import LEARNED_RECIPE, RGBDEncoder, pilot_binding
    from alexdoor_xas.policies.observations import B1Observer, require_provider_binding
    from alexdoor_xas.policies.pilot_runtime import (
        live_observation_binding,
        loaded_observation_recipe,
    )

    binding = pilot_binding(b1_binding.config["inspection"], recipe=LEARNED_RECIPE)
    recipe = PrototypeRecipe(
        json.dumps(
            dict(scan_fusion="object-v1", inspection=binding.config["inspection"], max_gap_s=0.15)
        )
    )
    assert live_observation_binding(binding, recipe) is binding
    loaded = loaded_observation_recipe(recipe, 1.0)
    live = live_observation_binding(binding, loaded)
    provider = SimpleNamespace(binding=loaded, generation=0)
    provider.reset = lambda: setattr(provider, "generation", provider.generation + 1)
    with pytest.raises(ValueError, match="inspection or freshness mismatch"):
        B1Observer(provider, binding, encoder=RGBDEncoder(binding, device="cpu"))
    observer = B1Observer(provider, live, encoder=RGBDEncoder(live, device="cpu"))
    require_provider_binding(provider, live)  # Same strict check used by PurdueIO.
    assert observer.binding == live
    original_config, live_config = binding.config, live.config
    assert live_config.pop("inspection") == loaded.config["inspection"]
    original_inspection = original_config.pop("inspection")
    assert live_config == original_config
    assert recipe.config["inspection"] == original_inspection
    points = np.asarray(loaded.config["inspection"]["waypoints"])
    np.testing.assert_array_equal(points[:, 1:], [points[0, 1:]] * 2)
    np.testing.assert_array_equal(points[:, 0], [0.0, 1.0])
    assert live.config["max_gap_s"] == binding.config["max_gap_s"]
    loaded_observation_recipe(recipe, 0.9)  # Seven real captures plus tracking margin.
    with pytest.raises(ValueError, match="all inspection captures"):
        loaded_observation_recipe(recipe, 0.899)

    changed = loaded.config
    changed["inspection"]["mount_pitch_rad"] += 0.01
    with pytest.raises(ValueError, match="cannot change inspection calibration"):
        live_observation_binding(binding, PrototypeRecipe(json.dumps(changed)))
    changed = loaded.config
    changed["max_gap_s"] = binding.config["max_gap_s"] + 0.01
    provider.binding = PrototypeRecipe(json.dumps(changed))
    with pytest.raises(ValueError, match="inspection or freshness mismatch"):
        B1Observer(provider, live, encoder=RGBDEncoder(live, device="cpu"))


def test_normal_acquisition_schedule_matches_teacher_provider_and_keeps_workspace(b1_binding):
    from copy import deepcopy

    from alexdoor_xas.perception.inspection import acquisition_inspection
    from alexdoor_xas.perception.provider import PrototypeRecipe
    from alexdoor_xas.policies.encoding import RGBDEncoder, pilot_binding
    from alexdoor_xas.policies.observations import B1Observer
    from alexdoor_xas.policies.pilot_runtime import (
        live_observation_binding,
        loaded_observation_recipe,
    )

    original = b1_binding.config["inspection"]
    acquisition = deepcopy(original)
    end, yaw, pitch = acquisition["waypoints"][-1]
    acquisition["waypoints"] += [[end + 2, yaw - 0.2, pitch], [end + 3, yaw - 0.2, pitch]]
    acquisition["sample_times_s"].append(end + 3)
    recipe = PrototypeRecipe(
        json.dumps(dict(scan_fusion="object-v1", inspection=original, max_gap_s=0.15))
    )
    live = loaded_observation_recipe(recipe, None, acquisition=acquisition)
    teacher = pilot_binding(acquisition_inspection(original, acquisition))
    binding = live_observation_binding(pilot_binding(original), live)
    assert teacher == binding
    provider = SimpleNamespace(binding=live, generation=0)
    provider.reset = lambda: setattr(provider, "generation", provider.generation + 1)
    B1Observer(provider, binding, encoder=RGBDEncoder(binding, device="cpu"))
    assert recipe.config["inspection"] == original
    assert live.config["inspection"] == acquisition
    acquisition["sample_times_s"][-1] += 1
    assert live.config["inspection"]["sample_times_s"][-1] == end + 3
