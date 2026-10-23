"""Saved measured-event replay through B1Runner; no physics or learned success."""

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from alexdoor_xas.action.spaces import A1_JOINT_DELTA
from alexdoor_xas.perception.admission import ActionAdmission, admit_action
from alexdoor_xas.perception.contracts import OPERATIONAL_V1
from alexdoor_xas.policies.act.execution import TemporalA1Execution, TemporalIngressObserver
from alexdoor_xas.policies.act.temporal import TemporalActPolicy
from alexdoor_xas.policies.common.b1_contract import PILOT_SCHEMA, PilotBinding
from alexdoor_xas.policies.observations import B1Observer, FeatureBlock, VisualEncoding
from alexdoor_xas.policies.rollout import B1Runner
from alexdoor_xas.policies.temporal import IncrementalObservationBuilder, TemporalConfig
from helpers import TEST_ROBOT_REF, NumericalIO, operational_provider
from perception_helpers import action_inputs, operational_estimate


class PredictionFixture:
    """Deterministic ordered output fixture; never represents learned behavior."""

    chunk_size, n_action_steps = 12, 6
    validate_snapshot = TemporalActPolicy.validate_snapshot

    def __init__(self, config):
        self.config, self.calls = config, []

    def reset(self, episode_id, generation):
        self.episode_id, self.generation = episode_id, generation
        self.calls.clear()

    def predict(self, snapshot, **kwargs):
        self.validate_snapshot(snapshot, now_s=kwargs["now_s"])
        start = len(self.calls) * 100
        self.calls.append(snapshot["decision_time_s"])
        return (start + np.arange(84).reshape(12, 7)) / 100000


@pytest.fixture
def replay_runner(tmp_path, b1_binding):
    release = b1_binding.to_dict()
    release["config"].update(geometry_profile=OPERATIONAL_V1, warmup_s=0.2)
    release["config"]["inspection"]["waypoints"] = [[0, 0, 0], [0.5, 0, 0]]
    release["config"]["inspection"]["sample_times_s"] = [0.5]
    release.update(schema=PILOT_SCHEMA, offline_passed=False, dynamic_passed=False, frozen=False)
    release["config"]["encoder"] = "numerical-fixture"
    binding = PilotBinding.from_dict(release)
    cfg = TemporalConfig(image_size=(2, 3), static_time_s=0.5)
    # Persist the sensory events separately from commands/geometry. On every
    # reset they are re-read as a new generation, never carried across resets.
    path = tmp_path / "measured-events.npz"
    np.savez(
        path,
        times=np.arange(121) / 60,
        q=np.zeros((121, 9)),
        dq=np.zeros((121, 9)),
        images=np.zeros((21, 5, 2, 3), "f4"),
    )
    saved = np.load(path)

    class Observer:
        generation = -1

        def reset(self):
            self.generation += 1

        def update(self, sensor):
            t = sensor["time_s"]
            return SimpleNamespace(
                time_s=t,
                frame=sensor["frame"],
                generation=self.generation,
                valid=True,
                features_available=True,
                reason="numerical_fixture",
                geometry_profile=OPERATIONAL_V1,
                features=np.zeros(binding.obs_dim),
                feature_support=VisualEncoding(
                    FeatureBlock(
                        np.zeros(2), min(t, 0.5), min(t, 0.5), sensor["frame"], self.generation
                    ),
                    FeatureBlock(np.zeros(2), t, t, sensor["frame"], self.generation),
                ),
                estimate=operational_estimate(t, self.generation),
            )

    class IO(NumericalIO):
        dt, max_ticks = 1 / 60, 120

        def reset(self):
            super().reset()
            self.pose = operational_estimate().operational.contact.world_pose
            self.q, self.applied = np.zeros(7), []

        def observe(self):
            sensor = super().observe()
            sensor["time_s"] = float(saved["times"][self.clock])
            return sensor

        def execute(self, command):
            # The existing numerical controller, with each row's measured q.
            import torch

            from alexdoor_xas.kinematics.pose_control import bounded_joint_step

            self.q += 0.001  # distinguish measured q from previous target/chunk anchor
            target, _ = bounded_joint_step(
                torch.tensor(command.value[None]),
                torch.tensor(self.q[None]),
                torch.tensor(np.tile([-2, 2], (7, 1))[None]),
                torch.ones(1, 7),
                self.dt,
            )
            self.applied.append((self.q.copy(), target.numpy()[0]))
            super().execute(command)

    observer = Observer()
    observer.binding = binding
    io = IO(binding)
    observations = IncrementalObservationBuilder(cfg, episode_id="saved-events")
    policy = PredictionFixture(cfg)
    execution = TemporalA1Execution(policy, binding, TEST_ROBOT_REF, observations)
    drop_after = [None]

    def events(sensor, observation):
        i, gen = sensor["frame"], observation.generation
        t = float(saved["times"][i])
        if drop_after[0] is not None and i > drop_after[0]:
            return []
        result = [
            dict(field=key, acquired=t, available=t, generation=gen, values=saved[key][i])
            for key in ("q", "dq")
        ]
        if i % 6 == 0:
            result.append(
                dict(
                    field="images",
                    acquired=t,
                    available=t,
                    generation=gen,
                    values=saved["images"][i // 6],
                    source_index=i,
                )
            )
        return result

    admissions = []

    def admission(observation, tool, row, remaining):
        proposal, evidence, limits = action_inputs(observation.estimate)
        proposal = replace(
            proposal,
            source="policy",
            action_space=A1_JOINT_DELTA,
            action=row,
            loaded=False,
            times_s=np.array([observation.time_s, observation.time_s + io.dt]),
        )
        evidence = tuple(replace(e, unloaded_separation_m=0.1) for e in evidence)
        result = admit_action(
            observation.estimate,
            proposal,
            evidence,
            limits,
            observation.time_s,
            generation=observation.generation,
        )
        assert result.admitted, result.reason
        admissions.append(row.copy())
        return result

    runner = B1Runner(
        execution,
        TemporalIngressObserver(observer, execution, events),
        io,
        admission_source=admission,
    )
    return runner, io, policy, execution, admissions, drop_after


def test_saved_events_order_six_of_twelve_replan_current_q_and_reset(replay_runner):
    runner, io, policy, execution, admissions, _ = replay_runner
    result = runner.run(max_ticks=13)
    assert result.reason == "time_budget" and result.control_steps == 13
    assert not result.predicted_complete
    expected = np.concatenate(
        [(100 * i + np.arange(84).reshape(12, 7))[:6] / 100000 for i in range(3)]
    )[:13]
    np.testing.assert_array_equal([c.value for c in io.commands], expected)
    np.testing.assert_array_equal(admissions, expected)
    for row, (q, applied) in zip(expected, io.applied, strict=True):
        np.testing.assert_allclose(applied, q + row, atol=1e-9)
    # Stop clears ingress/prediction state; a new run replays a new generation.
    assert not runner.pending and runner.source is None and execution.snapshot is None
    assert not any(execution.observations.rows.values())
    assert runner.run(max_ticks=7).control_steps == 7
    np.testing.assert_array_equal([c.value for c in io.commands], expected[:7])


def test_stale_current_state_stops_pending_chunk_before_replan(replay_runner):
    runner, io, _, _, _, drop_after = replay_runner
    drop_after[0] = 42  # inspection ends here, then lose current state
    result = runner.run(max_ticks=13)
    assert result.reason.startswith("invalid_execution:")
    assert "stale" in result.reason
    assert result.control_steps == 3 < 6 and len(io.commands) == 3
    assert not runner.pending and io.stops == 1


def test_admission_refusal_and_physical_stop_preserve_existing_stop_path(replay_runner):
    runner, io, _, _, _, _ = replay_runner
    original_admission = runner.admission_source
    runner.admission_source = lambda *args: ActionAdmission(False, False, "fixed_safety_refusal")
    assert runner.run(max_ticks=8).reason == "fixed_safety_refusal"
    assert not io.commands and not runner.pending
    io.fail_after = 2
    runner.admission_source = original_admission
    result = runner.run(max_ticks=8)
    assert result.reason == "force_limit" and result.control_steps == 2
    assert not runner.pending and io.stops == 1


def test_execution_rejects_changed_horizon_and_ensembling(replay_runner):
    _, _, _, execution, _, _ = replay_runner
    for options in (dict(n_action_steps=12), dict(temporal_ensemble=True)):
        with pytest.raises(ValueError, match="six"):
            execution.chunk_source(None, **options)


def test_preinitialized_provider_generation_drives_temporal_reset(replay_runner):
    runner, io, policy, execution, _, _ = replay_runner
    provider = operational_provider(execution.binding)
    # SessionProvider initializes/resets itself before B1Observer is created.
    # Its generation is independent of the observer's local reset counter.
    provider.reset()
    observer = B1Observer(provider, provider.binding)
    assert provider.generation != observer.generation
    execution.binding = io.binding = provider.binding
    runner.observer = TemporalIngressObserver(observer, execution, runner.observer.events)
    for ticks in (13, 7):
        result = runner.run(max_ticks=ticks)
        assert result.reason == "time_budget" and result.control_steps == ticks
        assert len(io.commands) == ticks and not result.predicted_complete
        assert policy.generation == execution.observations.generation == provider.generation
        assert provider.generation != observer.generation
        assert not runner.pending and not any(execution.observations.rows.values())


def test_intermediate_startup_skips_closed_door_inspection_and_clears_context(replay_runner):
    runner, io, policy, execution, _, _ = replay_runner
    cfg = replace(execution.observations.config, static_time_s=0)
    execution.observations = IncrementalObservationBuilder(cfg, episode_id="fresh-intermediate")
    with pytest.raises(ValueError, match="inspection recipe"):
        TemporalA1Execution(policy, execution.binding, TEST_ROBOT_REF, execution.observations)
    calls = []

    def startup(observer, io):
        calls.append(observer.generation)
        observation = observer.update(io.observe())
        for _ in range(12):  # new measured pre-roll, using numerical IO only
            if reason := io.safety_reason(None):
                return observation, reason
            io.hold_step()
            observation = observer.update(io.observe())
        return observation, ""

    execution.startup = startup
    # Constructor validates explicit local-t=0 producer rather than closed inspection.
    TemporalA1Execution(
        policy, execution.binding, TEST_ROBOT_REF, execution.observations, startup=startup
    )
    for _ in range(2):
        result = runner.run(max_ticks=7)
        assert result.reason == "time_budget" and result.control_steps == 7
        assert io.clock == 19  # .2 s fresh pre-roll + seven rows; no closed inspection
        assert not any(execution.observations.rows.values())
    assert len(set(calls)) == 2
    execution.startup = lambda observer, io: (observer.update(io.observe()), "unqualified_reset")
    result = runner.run(max_ticks=7)
    assert result.reason == "unqualified_reset" and not io.commands


def test_external_completion_ends_only_after_a_real_safe_policy_step(replay_runner):
    runner, io, *_ = replay_runner
    seen = []
    runner.completion_source = lambda: seen.append(len(io.commands)) or True
    result = runner.run(max_ticks=13)
    assert result.reason == "evaluated_task_complete" and result.control_steps == 1
    assert not result.predicted_complete
    assert seen == [1] and len(io.commands) == 1 and runner.stopped


def test_physical_safety_failure_has_priority_over_external_completion(replay_runner):
    runner, io, *_ = replay_runner
    seen = []
    runner.completion_source = lambda: seen.append(True) or True
    io.safety_reason = lambda stage: "forced_safety" if io.commands else ""
    result = runner.run(max_ticks=13)
    assert result.reason == "forced_safety" and result.control_steps == 1
    assert seen == [] and not result.predicted_complete and runner.stopped


def test_diagnostic_prefix_hands_off_once_with_live_ingress_and_never_calls_act_early(
    replay_runner,
):
    from alexdoor_xas.policies.act.execution import DiagnosticA1Prefix

    runner, io, policy, execution, _, _ = replay_runner
    actions = np.arange(30 * 7).reshape(30, 7) / 1000000
    execution.diagnostic_prefix = DiagnosticA1Prefix(0.7 + np.arange(30) / 60, actions, 6, io.dt)
    execute, sources, queries = io.execute, [], []
    predict = policy.predict

    def record_predict(snapshot, **options):
        queries.append(snapshot["decision_time_s"])
        return predict(snapshot, **options)

    policy.predict = record_predict

    def record(command):
        sources.append(execution.reference_command)
        execute(command)

    io.execute = record
    result = runner.run(max_ticks=13)
    assert result.reason == "time_budget" and result.control_steps == 13
    assert sources == [True] * 6 + [False] * 7
    np.testing.assert_array_equal([c.value for c in io.commands[:6]], actions[:6])
    np.testing.assert_allclose(queries, [0.8, 0.9], atol=1e-9)
    assert not any(execution.observations.rows.values())
    sources.clear()
    queries.clear()
    assert runner.run(max_ticks=7).control_steps == 7
    assert sources == [True] * 6 + [False]
    np.testing.assert_allclose(queries, [0.8], atol=1e-9)


def test_diagnostic_prefix_stale_live_input_stops_before_act_and_keeps_safety(replay_runner):
    from alexdoor_xas.policies.act.execution import DiagnosticA1Prefix

    runner, io, policy, execution, _, drop_after = replay_runner
    execution.diagnostic_prefix = DiagnosticA1Prefix(
        0.7 + np.arange(30) / 60, np.zeros((30, 7)), 6, io.dt
    )
    drop_after[0] = 42
    result = runner.run(max_ticks=13)
    assert result.reason.startswith("invalid_execution:") and "stale" in result.reason
    assert result.control_steps == 3 and not policy.calls and io.stops == 1


def test_diagnostic_prefix_rejects_bad_clock_and_discards_future_teacher_commands():
    from alexdoor_xas.policies.act.execution import DiagnosticA1Prefix

    times, actions = 25.2 + np.arange(30) / 60, np.ones((30, 7))
    for count in (0, 5, 30):
        with pytest.raises(ValueError, match="six-row"):
            DiagnosticA1Prefix(times, actions, count, 1 / 60)
    prefix = DiagnosticA1Prefix(times, actions, 6, 1 / 60)
    assert prefix.actions.shape == (6, 7) and len(prefix.times) == 7
    with pytest.raises(ValueError, match="clock"):
        prefix.chunk(25.21)
    np.testing.assert_array_equal(prefix.chunk(25.2), actions[:6])
    with pytest.raises(ValueError, match="clock"):
        prefix.chunk(25.31)
    assert prefix.chunk(25.3) is None
    assert prefix.chunk(100.0) is None and prefix.handed_off
