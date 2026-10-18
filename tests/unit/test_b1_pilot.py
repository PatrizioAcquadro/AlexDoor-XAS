"""Unqualified common policy plumbing; numerical encodings, no model or physics."""

import json
from dataclasses import replace
from types import SimpleNamespace

import h5py
import numpy as np
import pytest
import torch
from test_b1_observations import operational_provider, sample
from test_b1_rollout import NumericalIO

from alexdoor_xas.action.b1 import ACTION_DIMS
from alexdoor_xas.action.frames import ObjectFrame
from alexdoor_xas.action.spaces import A1_JOINT_DELTA, A2_EE_DELTA, A3_OBJ_REL_EE_DELTA
from alexdoor_xas.dataset.b1 import B1Dataset, compile_episode, export_dataset, validate_episode
from alexdoor_xas.perception.contracts import OPERATIONAL_V1
from alexdoor_xas.policies.act.config import ActModelCfg
from alexdoor_xas.policies.b1 import (
    B1Policy,
    batch_factories,
    load_b1_data,
    load_policy_payload,
    save_policy,
)
from alexdoor_xas.policies.common.b1_contract import (
    OBS_KEYS,
    PILOT_SCHEMA,
    PerceptionBinding,
    PilotBinding,
    policy_contract,
    validate_contract,
)
from alexdoor_xas.policies.diffusion.config import DiffusionModelCfg
from alexdoor_xas.policies.observations import (
    B1Observer,
    FeatureBlock,
    VisualEncoding,
    require_features,
)
from alexdoor_xas.policies.rollout import ActionAdapter, B1Runner
from helpers import TEST_ROBOT_REF


@pytest.fixture
def pilot_binding(b1_binding):
    config = b1_binding.config | {
        "encoder": "numerical-fixture",
        "geometry_profile": OPERATIONAL_V1,
    }
    config["inspection"]["waypoints"] = [[0, 0, 0], [1 / 60, 0, 0]]
    config["inspection"]["sample_times_s"] = [1 / 60]
    config["warmup_s"] = 0
    return PilotBinding.from_dict(
        dict(
            schema=PILOT_SCHEMA,
            config=config,
            offline_passed=False,
            dynamic_passed=False,
            frozen=False,
        )
    )


def sensor(t, frame):
    return sample(t, frame) | dict(
        rgb=np.ones((2, 2, 3), np.uint8),
        depth_m=np.ones((2, 2, 1)),
        valid_depth=np.ones((2, 2, 1), bool),
        annotations={"hinge": [9, 9, 9]},
        robot_feedback={"torque": np.ones(7)},
    )


class NumericalEncoder:
    def __init__(self, binding):
        self.binding = binding
        self.reset()

    def reset(self):
        self.calls = []
        self.between = False

    def update(self, value, *, generation):
        self.calls.append(value)
        if self.between:
            return None
        blocks = [
            FeatureBlock(
                np.arange(width), value["time_s"], value["time_s"], value["frame"], generation
            )
            for width in self.binding.config["visual_dims"]
        ]
        return VisualEncoding(*blocks)


def pilot_episode(binding):
    observer = B1Observer(None, binding, encoder=NumericalEncoder(binding))
    observations = [observer.update(sensor(i / 60, i)) for i in range(3)]
    tool = ObjectFrame(np.zeros(3), np.eye(3))
    return compile_episode(
        episode_id="pilot",
        asset_id="left",
        asset_splits={"left": "train"},
        observations=observations,
        tools=[tool] * 3,
        goals=[tool] * 2,
        joint_targets=np.ones((2, 7)) * 0.01,
        binding=binding,
    )


def test_pilot_features_do_not_require_hinge_or_leak_labels(pilot_binding):
    encoder = NumericalEncoder(pilot_binding)
    observer = B1Observer(None, pilot_binding, encoder=encoder)
    first = observer.update(sensor(0, 0))
    assert first.features_available and not first.geometry_available and not first.valid
    assert first.estimate.frame is None and not first.estimate.valid
    assert set(encoder.calls[-1]) == {
        "time_s",
        "frame",
        "rgb",
        "depth_m",
        "valid_depth",
        "joint_position",
        "joint_velocity",
    }
    encoder.between = True
    next_sensor = sensor(1 / 60, 1)
    next_sensor["joint_position"][:] = 0.2
    current = observer.update(next_sensor)
    np.testing.assert_array_equal(current.features[-18:-9], 0.2)
    assert current.feature_support.recent.acquired_s == 0
    assert current.feature_support.recent.available_s == 0
    assert not replace(current, time_s=0.16).features_available
    assert observer.update(sensor(0.16, 2)).reason == "stale_policy_features"
    observer.reset()
    encoder.between = True
    assert not observer.update(sensor(0, 0)).features_available


@pytest.mark.parametrize("defect", ["future", "generation", "width", "nan", "frame"])
def test_feature_support_rejects_future_cross_episode_and_invalid_blocks(pilot_binding, defect):
    observer = B1Observer(None, pilot_binding, encoder=NumericalEncoder(pilot_binding))
    obs = observer.update(sensor(0, 0))
    block = obs.feature_support.recent
    changes = {
        "future": dict(available_s=0.01),
        "generation": dict(generation=obs.generation + 1),
        "width": dict(values=np.zeros(3)),
        "nan": dict(values=np.array([0, np.nan])),
        "frame": dict(frame=1),
    }
    bad = replace(obs.feature_support, recent=replace(block, **changes[defect]))
    with pytest.raises(ValueError):
        require_features(replace(obs, feature_support=bad), pilot_binding)


def test_geometry_and_feature_availability_are_independent(pilot_binding):
    provider = operational_provider(pilot_binding)
    provider.binding = pilot_binding
    encoder = NumericalEncoder(pilot_binding)
    observer = B1Observer(provider, pilot_binding, encoder=encoder)
    first = observer.update(sensor(0, 0))
    assert first.features_available and first.geometry_available
    provider.between = encoder.between = True
    held = observer.update(sensor(0.1, 1))
    assert held.geometry_available and held.estimate is first.estimate
    assert held.estimate.operational.angle_support.supported_s == 0
    # New visual features cannot renew the old geometry's support timestamp.
    encoder.between = False
    stale = observer.update(sensor(0.16, 2))
    assert stale.features_available and not stale.geometry_available
    provider.between = False
    encoder.between = True
    unavailable = observer.update(sensor(0.32, 3))
    assert not unavailable.features_available and unavailable.geometry_available


@pytest.mark.parametrize("space", [A1_JOINT_DELTA, A2_EE_DELTA])
def test_robot_command_interpretation_does_not_admit_motion(pilot_binding, space):
    adapter = ActionAdapter(space, 0.15, profile=OPERATIONAL_V1)
    row = np.zeros(ACTION_DIMS[space])
    assert adapter.interpret(row).admission is None
    observation = B1Observer(None, pilot_binding, encoder=NumericalEncoder(pilot_binding)).update(
        sensor(0, 0)
    )
    with pytest.raises(ValueError):
        adapter.command(
            observation, ObjectFrame(np.zeros(3), np.eye(3)), action=row, remaining_ticks=1
        )
    with pytest.raises(ValueError):
        ActionAdapter(A3_OBJ_REL_EE_DELTA, 0.15).interpret(np.zeros(6))


def test_partial_data_cannot_become_matched_or_invent_object_actions(
    tmp_path, pilot_binding, b1_binding
):
    episode = pilot_episode(pilot_binding)
    assert set(episode.actions) == {A1_JOINT_DELTA, A2_EE_DELTA}
    root = export_dataset(
        [episode], tmp_path / "pilot", pilot_binding, TEST_ROBOT_REF, {"left": "train"}
    )
    assert B1Dataset(root, A1_JOINT_DELTA).episode_ids == ["pilot"]
    with h5py.File(root / "episode_000000.hdf5") as source:
        assert A3_OBJ_REL_EE_DELTA not in source
    with pytest.raises(ValueError, match="unavailable"):
        B1Dataset(root, A3_OBJ_REL_EE_DELTA)
    with pytest.raises(ValueError, match="A1-A4"):
        validate_episode(episode, b1_binding, {"left": "train"})
    metadata = json.loads((root / "meta.json").read_text())
    metadata["schema"] = "b1.policy-dataset.v1"
    (root / "meta.json").write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="Unsupported"):
        B1Dataset(root, A1_JOINT_DELTA)


@pytest.mark.parametrize("key", ["offline_passed", "dynamic_passed", "frozen"])
def test_pilot_never_weakens_release_validation(pilot_binding, b1_binding, key):
    value = pilot_binding.to_dict()
    value[key] = True
    with pytest.raises(ValueError, match="unqualified"):
        PilotBinding.from_dict(value)
    with pytest.raises(ValueError, match="qualified and frozen"):
        PerceptionBinding.from_dict(pilot_binding.to_dict())
    with pytest.raises(ValueError, match="mismatch"):
        validate_contract(policy_contract(pilot_binding, A1_JOINT_DELTA), b1_binding)


@pytest.mark.parametrize("family", ["act", "diffusion"])
@pytest.mark.parametrize("space", [A1_JOINT_DELTA, A2_EE_DELTA])
def test_pilot_uses_existing_data_policy_checkpoint_and_stops_before_execution(
    tmp_path, pilot_binding, b1_binding, family, space
):
    root = export_dataset(
        [pilot_episode(pilot_binding)],
        tmp_path / "pilot",
        pilot_binding,
        TEST_ROBOT_REF,
        {"left": "train"},
    )
    data = load_b1_data(root, space, binding=pilot_binding)
    train, _ = batch_factories(data, family, 2, 1, 0)
    assert next(iter(train(0)))["actions"].shape == (1, 2, ACTION_DIMS[space])
    cfg = ActModelCfg() if family == "act" else DiffusionModelCfg()
    envelope = SimpleNamespace(
        obs_dim=data.obs_dim,
        action_dim=data.action_dim,
        cfg=cfg,
        state_dict=lambda: {"fixture": torch.ones(1)},
    )
    path = save_policy(tmp_path / "policy.pt", family, envelope, data)
    _, contract = load_policy_payload(
        path, family, binding=pilot_binding, runtime_asset=TEST_ROBOT_REF
    )
    with pytest.raises(ValueError, match="mismatch"):
        load_policy_payload(path, family, binding=b1_binding, runtime_asset=TEST_ROBOT_REF)
    calls = []
    numerical = SimpleNamespace(
        stats=data.stats,
        chunk_size=2,
        seed=lambda _: None,
        predict=lambda features: calls.append(features) or np.zeros((2, ACTION_DIMS[space])),
    )
    policy = B1Policy(numerical, family, contract, robot_asset=TEST_ROBOT_REF)
    io = NumericalIO(pilot_binding)
    io.dt = 1 / 60
    io.observe = lambda: sensor(io.clock * io.dt, io.clock)
    observer = B1Observer(None, pilot_binding, encoder=NumericalEncoder(pilot_binding))
    runner = B1Runner(policy, observer, io, pilot_inference_only=True)
    result = runner.run(max_ticks=1)
    assert result.reason == "unqualified_pilot_inference_only" and len(calls) == 1
    assert not io.commands and io.stops == 1 and not result.predicted_complete
    runner = B1Runner(policy, observer, io)
    result = runner.run(max_ticks=1)
    assert result.reason == "missing_action_admission_source" and not io.commands
    runner = B1Runner(policy, observer, io, admission_source=lambda *args: None)
    result = runner.run(max_ticks=1)
    assert result.reason.startswith("invalid_execution") and not io.commands


@pytest.mark.parametrize("defect", ["mask", "depth", "alignment", "proprioception", "time"])
def test_bad_acquisition_cannot_reuse_pilot_features(pilot_binding, defect):
    encoder = NumericalEncoder(pilot_binding)
    observer = B1Observer(None, pilot_binding, encoder=encoder)
    assert observer.update(sensor(0, 0)).features_available
    encoder.between = True
    value = sensor(1 / 60, 1)
    if defect == "mask":
        value["valid_depth"][:] = False
    elif defect == "depth":
        value["depth_m"][0, 0, 0] = np.nan
    elif defect == "alignment":
        value["rgb"] = np.ones((1, 1, 3))
    elif defect == "proprioception":
        value["joint_position"][0] = np.inf
    else:
        value["time_s"] = 0
    assert not observer.update(value).features_available


def test_partial_data_keeps_pilot_control_cadence(pilot_binding):
    episode = pilot_episode(pilot_binding)
    with pytest.raises(ValueError, match="60 Hz"):
        validate_episode(
            replace(episode, times=episode.times * 2), pilot_binding, {"left": "train"}
        )


@pytest.mark.parametrize(
    "physics_dt,decimation,control_dt",
    [(1 / 60, 2, 1 / 60), (1 / 120, 1, 1 / 60), (1 / 120, 2, 1 / 30)],
)
def test_pilot_io_rejects_cadence_changes_before_runtime_setup(
    pilot_binding, physics_dt, decimation, control_dt
):
    from alexdoor_xas.policies.purdue import PurdueIO

    env = SimpleNamespace(
        cfg=SimpleNamespace(sim=SimpleNamespace(dt=physics_dt), decimation=decimation),
        step_dt=control_dt,
    )
    with pytest.raises(ValueError, match="120 Hz physics and 60 Hz control"):
        PurdueIO(env, binding=pilot_binding, robot_asset=TEST_ROBOT_REF, setup=None)


def test_runner_passes_only_predicted_action_to_admission_source(tmp_path, pilot_binding):
    from test_operational_admission import action_inputs

    from alexdoor_xas.perception.admission import admit_action

    root = export_dataset(
        [pilot_episode(pilot_binding)],
        tmp_path / "pilot",
        pilot_binding,
        TEST_ROBOT_REF,
        {"left": "train"},
    )
    data = load_b1_data(root, A2_EE_DELTA)
    row = np.array([0.001, 0, 0, 0, 0, 0])
    numerical = SimpleNamespace(stats=data.stats, chunk_size=1, predict=lambda _: row[None])
    policy = B1Policy(numerical, "act", data.dataset.contract, robot_asset=TEST_ROBOT_REF)
    provider = operational_provider(pilot_binding)
    provider.binding = pilot_binding
    observer = B1Observer(provider, pilot_binding, encoder=NumericalEncoder(pilot_binding))
    io = NumericalIO(pilot_binding)
    io.dt = 1 / 60
    reset_io = io.reset

    def reset():
        reset_io()
        io.pose = ObjectFrame(np.array([0, 0.3, 1]), np.eye(3))

    io.reset = reset
    io.observe = lambda: sensor(io.clock * io.dt, io.clock)
    calls = []

    def admit(observation, tool, action, remaining):
        np.testing.assert_array_equal(action, row)
        proposal, evidence, limits = action_inputs(observation.estimate)
        proposal = replace(
            proposal,
            source="policy",
            loaded=False,
            times_s=np.array([observation.time_s, observation.time_s + io.dt]),
        )
        evidence = tuple(replace(e, unloaded_separation_m=0.1) for e in evidence)
        receipt = admit_action(
            observation.estimate,
            proposal,
            evidence,
            limits,
            observation.time_s,
            generation=observation.generation,
        )
        calls.append(receipt)
        return receipt

    result = B1Runner(policy, observer, io, admission_source=admit).run(max_ticks=1)
    assert result.reason == "time_budget" and result.control_steps == 1
    assert len(calls) == len(io.commands) == 1 and calls[0].admitted
    assert io.commands[0].admission is calls[0]  # numerical routing, no physics/contact


def test_raw_pilot_preparation_uses_causal_encoder_and_robot_fk(tmp_path, pilot_binding):
    from test_b1_recording import obs

    from alexdoor_xas.assets.purdue import ARM_JOINTS, NECK_JOINTS
    from alexdoor_xas.dataset.b1 import PILOT_PURPOSE, prepare_recording
    from alexdoor_xas.recording.b1 import B1Writer

    path = tmp_path / "raw.hdf5"
    writer = B1Writer(
        path,
        dict(
            episode_id="raw-pilot",
            asset_id="left",
            split="train",
            condition="nominal",
            control_dt=1 / 60,
            purpose=PILOT_PURPOSE,
            inspection=pilot_binding.config["inspection"],
        ),
        dict(depth_interval_m=[0.1, 5], joint_names=list(ARM_JOINTS + NECK_JOINTS)),
    )
    tool = ObjectFrame(np.zeros(3), np.eye(3))
    writer.observe(obs(), {})
    for i in range(2):
        writer.transition(
            dict(
                time_s=i / 60,
                joint_target=np.zeros(7),
                tool_position=tool.origin,
                tool_rotation=tool.rot,
                phase=0,
            ),
            obs((i + 1) / 60, i + 2),
            {},
        )
    writer.finish(dict(passed=True, termination="safe_release"))
    writer.close()
    observer = B1Observer(None, pilot_binding, encoder=NumericalEncoder(pilot_binding))
    episode = prepare_recording(path, observer, lambda q, calibration: tool, {"left": "train"})
    assert set(episode.actions) == {A1_JOINT_DELTA, A2_EE_DELTA}
    assert episode.observations.shape == (3, pilot_binding.obs_dim)


def test_pilot_geometry_recipe_is_independent_but_inspection_and_age_must_match(pilot_binding):
    from alexdoor_xas.perception.provider import PrototypeRecipe

    provider = operational_provider(pilot_binding)
    config = {
        "scan_fusion": "object-v1",
        "inspection": pilot_binding.config["inspection"],
        "max_gap_s": pilot_binding.config["max_gap_s"],
    }
    provider.binding = PrototypeRecipe(json.dumps(config))
    observer = B1Observer(provider, pilot_binding, encoder=NumericalEncoder(pilot_binding))
    observation = observer.update(sensor(0, 0))
    assert observation.features_available and observation.geometry_available
    assert not provider.binding.to_dict()["dynamic_passed"]
    for bad in (config | {"inspection": {}}, config | {"max_gap_s": 0.2}):
        provider.binding = PrototypeRecipe(json.dumps(bad))
        with pytest.raises(ValueError, match="inspection or freshness mismatch"):
            B1Observer(provider, pilot_binding, encoder=NumericalEncoder(pilot_binding))


def test_policy_views_and_chunk_sampler_share_readonly_disk_matrix(pilot_binding, tmp_path):
    from alexdoor_xas.dataset.observations import obs_matrix
    from alexdoor_xas.dataset.sampling import ChunkSampler

    root = export_dataset(
        [pilot_episode(pilot_binding)],
        tmp_path / "data",
        pilot_binding,
        TEST_ROBOT_REF,
        {"left": "train"},
    )
    dataset = B1Dataset(root, A1_JOINT_DELTA)
    record = dataset.records[0]
    assert isinstance(record.observation_matrix, np.memmap)
    assert not record.observation_matrix.flags.writeable
    assert np.shares_memory(obs_matrix(record, OBS_KEYS), record.observation_matrix)
    sampler = ChunkSampler(dataset, 1, obs_keys=OBS_KEYS)
    assert np.shares_memory(sampler._obs[0], record.observation_matrix)
    # Explicit replacement of a column cannot be hidden by the canonical fast path.
    record.obs[OBS_KEYS[-1]] = np.ones_like(record.obs[OBS_KEYS[-1]])
    assert np.all(obs_matrix(record, OBS_KEYS)[:, -9:] == 1)


def test_excluded_supervision_is_not_required_on_training_host(tmp_path):
    import runpy
    from pathlib import Path

    prepare = runpy.run_path(str(Path("scripts/pilot_offline.py")))["prepare"]
    entry = dict(asset_id="retained", split="train", use_for_imitation=False)
    with pytest.raises(ValueError, match="explicit reason"):
        prepare(tmp_path, {"episodes": [entry]}, export=False)
    entry["imitation_exclusion_reason"] = "Preserved on the collection host; unsuitable labels"
    prepare(tmp_path, {"episodes": [entry]}, export=False)
    audit = json.loads((tmp_path / "preflight-label-audit.json").read_text())
    assert audit[0]["used_for_imitation"] is False
    assert audit[0]["recording_available_here"] is False
    assert audit[0]["retention_verified_here"] is False


def test_remote_train_only_path_never_substitutes_train_for_missing_development():
    import runpy
    from pathlib import Path

    metrics = runpy.run_path(str(Path("scripts/pilot_offline.py")))["raw_metrics"]
    assert metrics(None, SimpleNamespace(val_ids=()), {}) == dict(
        evaluated=False, reason="development_records_not_on_this_host"
    )


def test_pilot_training_state_preserves_payload_and_rejects_changed_experiment(tmp_path):
    import runpy

    api = runpy.run_path("scripts/pilot_offline.py")
    # Serialization only: no CPU model training or inference.
    model = torch.nn.Linear(2, 1)
    path = tmp_path / "training-state.pt"
    protocol = dict(training_seed=17, epochs=30, max_batches_per_epoch=100)
    state = dict(
        global_step=100,
        next_epoch=1,
        optimizer_state={"step": 100},
        rng_states={"cpu": torch.get_rng_state()},
        ema_state={"updates": 100},
    )
    rows = [dict(epoch=0, batch_count=100)]
    weights = {k: v.clone() for k, v in model.state_dict().items()}
    api["save_training_state"](path, model, state, protocol, "act", "A1_joint_delta", rows)
    assert path.is_file() and not path.with_suffix(".tmp").exists()
    with torch.no_grad():
        model.weight.zero_()
    restored, epochs = api["restore_training_state"](
        path,
        model,
        protocol,
        "act",
        "A1_joint_delta",
    )
    assert restored["global_step"] == 100 and restored["ema_state"] == {"updates": 100}
    assert epochs == rows
    for name, value in weights.items():
        torch.testing.assert_close(model.state_dict()[name], value, rtol=0, atol=0)
    for changed, family, space in (
        ({**protocol, "training_seed": 18}, "act", "A1_joint_delta"),
        (protocol, "diffusion", "A1_joint_delta"),
        (protocol, "act", "A2_ee_delta"),
    ):
        with pytest.raises(ValueError, match="differs"):
            api["restore_training_state"](path, model, changed, family, space)


def test_pilot_continuation_allows_only_epoch_increase(tmp_path):
    import runpy

    api = runpy.run_path("scripts/pilot_offline.py")
    source = dict(epochs=30, max_batches_per_epoch=100, overfit_updates=0, training_seed=17)
    target = source | dict(epochs=44)
    api["validate_epoch_extension"](source, target)
    for changed in (
        source,
        source | dict(epochs=29),
        target | dict(training_seed=18),
        target | dict(max_batches_per_epoch=90),
        target | dict(overfit_updates=1),
        target | dict(extra_recipe="changed"),
    ):
        with pytest.raises(ValueError, match="strict epoch increase"):
            api["validate_epoch_extension"](source, changed)
    api["write_json"](tmp_path / "protocol.json", source)
    report = {}
    for space in ("A1_joint_delta", "A2_ee_delta"):
        for family in ("act", "diffusion"):
            name = f"{family}-{space}"
            directory = tmp_path / name
            directory.mkdir()
            # File existence/protocol guard only, no inference or optimization.
            (directory / "final.pt").touch()
            (directory / "training-state.pt").touch()
            report[name] = dict(optimizer_updates=3000)
    api["write_json"](tmp_path / "report.json", report)
    before = (tmp_path / "protocol.json").read_bytes()
    assert api["continuation_source"](tmp_path, target) == dict(
        directory=str(tmp_path.resolve()), protocol=source
    )
    assert (tmp_path / "protocol.json").read_bytes() == before
    (tmp_path / "diffusion-A2_ee_delta/training-state.pt").unlink()
    with pytest.raises(ValueError, match="four complete"):
        api["continuation_source"](tmp_path, target)


def test_completed_resume_validates_reports_without_reopening_feature_data(tmp_path, monkeypatch):
    import runpy

    api = runpy.run_path("scripts/pilot_offline.py")
    destination = tmp_path / "training"
    destination.mkdir()
    protocol = dict(epochs=44, max_batches_per_epoch=100, overfit_updates=0)
    api["write_json"](destination / "protocol.json", protocol)
    report = {}
    for space in ("A1_joint_delta", "A2_ee_delta"):
        for family in ("act", "diffusion"):
            name = f"{family}-{space}"
            (destination / name).mkdir()
            # Metadata/existence guard only; these files are never loaded as models.
            (destination / name / "final.pt").touch()
            report[name] = dict(optimizer_updates=4400)
    api["write_json"](destination / "report.json", report)

    def forbidden_loader(*args, **kwargs):
        raise AssertionError("Completed cells must not reopen feature data")

    monkeypatch.setitem(api["train"].__globals__, "load_b1_data", forbidden_loader)
    assert api["train"](tmp_path, protocol, resume=True) == report
    (destination / "act-A1_joint_delta/final.pt").unlink()
    with pytest.raises(ValueError, match="Incomplete reported final cell"):
        api["train"](tmp_path, protocol, resume=True)


def test_partial_resume_skips_only_complete_representations(tmp_path, monkeypatch):
    import runpy

    api = runpy.run_path("scripts/pilot_offline.py")
    destination = tmp_path / "training"
    destination.mkdir()
    protocol = dict(epochs=44, max_batches_per_epoch=100, overfit_updates=0)
    api["write_json"](destination / "protocol.json", protocol)
    report = {}
    for family in ("act", "diffusion"):
        name = f"{family}-A1_joint_delta"
        (destination / name).mkdir()
        (destination / name / "final.pt").touch()
        report[name] = dict(optimizer_updates=4400)
    api["write_json"](destination / "report.json", report)
    called = []

    def loader(directory, space):
        called.append(space)
        raise RuntimeError("unfinished data reached")

    monkeypatch.setitem(api["train"].__globals__, "load_b1_data", loader)
    monkeypatch.setitem(api["train"].__globals__, "dataset_ram_bytes", lambda _: 0)
    monkeypatch.setitem(api["train"].__globals__, "memory_checkpoint", lambda *a, **k: None)
    with pytest.raises(RuntimeError, match="unfinished data reached"):
        api["train"](tmp_path, protocol, resume=True)
    assert called == ["A2_ee_delta"]


def test_append_only_train_continuation_preserves_exclusions_and_frozen_fit():
    import runpy

    validate = runpy.run_path("scripts/pilot_offline.py")["validate_epoch_extension"]
    old = dict(episode_id="old", asset_id="left", split="train")
    excluded = dict(episode_id="excluded", asset_id="left", split="train", use_for_imitation=False)
    added = dict(episode_id="new", asset_id="left", split="train")
    source = dict(
        epochs=240,
        training_seed=17,
        lr=0.0001,
        overfit_updates=0,
        episodes=[old, excluded],
        train_sampling="phase_balanced_v1",
    )
    target = source | dict(
        epochs=260, episodes=[old, excluded, added], normalization_episode_ids=["old"]
    )
    validate(source, target)
    for change in (
        target | dict(training_seed=18),
        target | dict(lr=0.0002),
        target | dict(normalization_episode_ids=["old", "new"]),
        target | dict(episodes=[old, excluded | dict(use_for_imitation=True), added]),
        target | dict(episodes=[old, excluded, added | dict(split="development")]),
        target | dict(episodes=[old, excluded, added | dict(asset_id="unseen")]),
        target | dict(episodes=[old, excluded, added, added]),
    ):
        with pytest.raises(ValueError, match="strict epoch increase"):
            validate(source, change)


def test_materialized_train_data_keeps_original_fit_and_file_backing(tmp_path, pilot_binding):
    from alexdoor_xas.dataset.b1 import materialize_train_data
    from alexdoor_xas.policies.b1 import validate_data

    original = pilot_episode(pilot_binding)
    original = replace(original, observations=original.observations.astype("f4").astype("f8"))
    new = replace(
        original,
        episode_id="new",
        observations=original.observations + 1000,
        actions={key: value + 1 for key, value in original.actions.items()},
    )
    old_root = export_dataset(
        [original], tmp_path / "old", pilot_binding, TEST_ROBOT_REF, {original.asset_id: "train"}
    )
    new_root = export_dataset(
        [new], tmp_path / "new", pilot_binding, TEST_ROBOT_REF, {new.asset_id: "train"}
    )
    before = load_b1_data(old_root, A1_JOINT_DELTA).stats.to_dict()
    destination = materialize_train_data(
        [old_root, new_root], tmp_path / "flat", normalization_episode_ids=[original.episode_id]
    )
    data = load_b1_data(destination, A1_JOINT_DELTA)
    assert data.train_ids == (original.episode_id, "new")
    assert data.stats.to_dict() == before
    assert isinstance(data.dataset.records[0].observation_matrix, np.memmap)
    assert data.dataset.records[0].observation_matrix.dtype == np.dtype("f4")
    validate_data(data)
    with h5py.File(destination / "episode_000001.hdf5") as file:
        np.testing.assert_array_equal(file["observations"][:], new.observations)
        np.testing.assert_array_equal(
            file[A1_JOINT_DELTA + "/actions"][:], new.actions[A1_JOINT_DELTA]
        )
    metadata = json.loads((destination / "meta.json").read_text())
    metadata["normalization_episode_ids"] = ["development"]
    (destination / "meta.json").write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="train-only"):
        load_b1_data(destination, A1_JOINT_DELTA)


def test_materialization_rejects_inexact_conversion_and_development(tmp_path, pilot_binding):
    from alexdoor_xas.dataset.b1 import materialize_train_data

    episode = pilot_episode(pilot_binding)
    values = episode.observations.copy()
    values[0, 0] = 0.1  # not an exact float32 value
    episode = replace(episode, observations=values)
    source = export_dataset(
        [episode], tmp_path / "source", pilot_binding, TEST_ROBOT_REF, {episode.asset_id: "train"}
    )
    with pytest.raises(ValueError, match="exactly float32"):
        materialize_train_data(
            [source], tmp_path / "failed", normalization_episode_ids=[episode.episode_id]
        )
    assert not (tmp_path / "failed").exists()
    assert list(tmp_path.glob(".b1-materialize-*"))  # failed staging retained
    with h5py.File(source / "episode_000000.hdf5") as file:
        assert file["observations"][0, 0] == 0.1
    development = replace(episode, split="development")
    source_dev = export_dataset(
        [development],
        tmp_path / "dev",
        pilot_binding,
        TEST_ROBOT_REF,
        {episode.asset_id: "development"},
    )
    with pytest.raises(ValueError, match="Development/test"):
        materialize_train_data(
            [source_dev], tmp_path / "failed-dev", normalization_episode_ids=[episode.episode_id]
        )
