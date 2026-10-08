"""Eight numerical dispatch paths and failure stops; not physical rollouts."""

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from alexdoor_xas.action.b1 import (
    ACTION_DIMS,
    STAGES,
    Segment,
    SegmentMotion,
    apply_pose_delta,
    pose_delta,
)
from alexdoor_xas.action.frames import ObjectFrame
from alexdoor_xas.action.spaces import A1_JOINT_DELTA, A2_EE_DELTA, A4_OBJ_CENTRIC_CHUNK
from alexdoor_xas.dataset.b1 import export_dataset
from alexdoor_xas.policies.b1 import B1Policy, load_b1_data
from alexdoor_xas.policies.observations import B1Observer, PolicyObservation
from alexdoor_xas.policies.purdue import PurdueIO, contact_status
from alexdoor_xas.policies.rollout import ActionAdapter, B1Runner, Command, MatchedReplay
from conftest import TEST_ROBOT_REF, make_b1_episode
from test_b1_observations import fake_estimator, sample
from test_operational_admission import action_inputs, operational_estimate


def operational_observation(estimate, time=0):
    from alexdoor_xas.perception.contracts import OPERATIONAL_V1

    return PolicyObservation(
        time, 1, np.zeros(22), estimate, "observed", OPERATIONAL_V1, estimate.operational.generation
    )


def test_provisional_adapter_requires_exact_diagnostic_action_and_explicit_admission():
    from alexdoor_xas.action.spaces import A2_EE_DELTA
    from alexdoor_xas.perception.admission import admit_action
    from alexdoor_xas.perception.contracts import OPERATIONAL_V1

    estimate = operational_estimate(count=2)
    observation = operational_observation(estimate)
    proposal, evidence, limits = action_inputs(estimate)
    admission = admit_action(estimate, proposal, evidence, limits, 0, generation=0)
    adapter = ActionAdapter(A2_EE_DELTA, 0.15, profile=OPERATIONAL_V1, source="diagnostic")
    command = adapter.command(
        observation,
        proposal.world_poses[0],
        action=proposal.action,
        remaining_ticks=1,
        admission=admission,
    )
    np.testing.assert_array_equal(command.value, proposal.action)
    policy = ActionAdapter(A2_EE_DELTA, 0.15, profile=OPERATIONAL_V1)
    with pytest.raises(ValueError):
        policy.command(
            observation,
            proposal.world_poses[0],
            action=proposal.action,
            remaining_ticks=1,
            admission=admission,
        )
    adapter.reset()
    with pytest.raises(ValueError, match="differs from admitted action"):
        adapter.command(
            observation,
            proposal.world_poses[0],
            action=proposal.action + 0.001,
            remaining_ticks=1,
            admission=admission,
        )
    with pytest.raises(ValueError, match="explicit reset"):
        adapter.command(
            observation,
            proposal.world_poses[0],
            action=proposal.action,
            remaining_ticks=1,
            admission=admission,
        )


def test_operational_a4_keeps_admitted_frame_and_latches_on_incompatible_updates():
    from alexdoor_xas.perception.admission import admit_action
    from alexdoor_xas.perception.contracts import OPERATIONAL_V1

    estimate = operational_estimate(count=2)
    proposal, evidence, limits = action_inputs(estimate)
    tool = proposal.world_poses[0]
    target = ObjectFrame(tool.origin + [0.001, 0, 0], tool.rot)
    segment = Segment("approach", target, 0.002, 2)
    motion = SegmentMotion.start(segment, tool, estimate.frame, estimate.signed_angle)
    poses = (tool, motion.goal(1, estimate.frame), motion.goal(2, estimate.frame))
    proposal = replace(
        proposal,
        action_space=A4_OBJ_CENTRIC_CHUNK,
        action=segment.encode(),
        world_poses=poses,
        times_s=np.array([0, 0.05, 0.1]),
    )
    evidence = tuple(replace(e, world_poses=poses) for e in evidence)
    receipt = admit_action(estimate, proposal, evidence, limits, 0, generation=0)
    assert receipt.admitted
    adapter = ActionAdapter(A4_OBJ_CENTRIC_CHUNK, 0.15, profile=OPERATIONAL_V1, source="diagnostic")
    adapter.command(
        operational_observation(estimate),
        tool,
        action=segment.encode(),
        remaining_ticks=2,
        admission=receipt,
    )
    current = operational_estimate(time=0.05, count=2)
    state = current.operational
    refined = ObjectFrame(state.hypotheses[0].frame.origin + [0.0005, 0, 0], np.eye(3))
    world_contact = ObjectFrame(refined.point_to_world(state.contact.local_pose.origin), np.eye(3))
    state = replace(
        state,
        hypotheses=(replace(state.hypotheses[0], frame=refined), state.hypotheses[1]),
        contact=replace(state.contact, selected_s=0, world_pose=world_contact),
    )
    current = replace(
        current, frame=refined, contact_position=world_contact.origin, operational=state
    )
    command = adapter.command(operational_observation(current, 0.05), poses[1], remaining_ticks=1)
    np.testing.assert_allclose(command.value.origin, poses[2].origin, atol=1e-12)
    np.testing.assert_allclose(command.value.rot, poses[2].rot, atol=1e-12)
    assert not adapter.done  # no invented release or termination
    adapter.reset()
    adapter.command(
        operational_observation(estimate),
        tool,
        action=segment.encode(),
        remaining_ticks=2,
        admission=receipt,
    )
    wrong = replace(current, operational=replace(state, leaf_id="different-leaf"))
    with pytest.raises(ValueError, match="reference_changed"):
        adapter.command(operational_observation(wrong, 0.05), poses[1], remaining_ticks=1)
    assert adapter.motion is None and adapter.admission is None
    with pytest.raises(ValueError, match="explicit reset"):
        adapter.command(operational_observation(current, 0.05), poses[1], remaining_ticks=1)


def test_missing_feedback_blocks_operational_io_before_physics_or_helpful_commands():
    from alexdoor_xas.perception.admission import admit_action
    from alexdoor_xas.perception.contracts import OPERATIONAL_V1
    from alexdoor_xas.perception.control import ObservedControlChecks

    estimate = operational_estimate()
    proposal, evidence, limits = action_inputs(estimate)
    receipt = admit_action(estimate, proposal, evidence, limits, 0, generation=0)
    io = PurdueIO.__new__(PurdueIO)
    io.profile = OPERATIONAL_V1
    io.env = SimpleNamespace(capture=SimpleNamespace(sample=SimpleNamespace(time_s=0)))
    feedback = io.robot_feedback()
    assert feedback.torque_nm is None
    sensor = sample(0, 1)
    sensor.update(rgb=np.ones((1, 1, 3)), valid_depth=np.ones((1, 1), bool))
    checks = ObservedControlChecks(SimpleNamespace(orientation_tolerance=0.1))
    reason = checks.check(
        sensor,
        estimate,
        proposal.world_poses[0],
        np.tile([-2, 2], (7, 1)),
        "push",
        np.zeros(2),
        profile=OPERATIONAL_V1,
        feedback=feedback,
        admission=receipt,
    )
    assert reason == "force_feedback_unavailable"
    assert (
        checks.check(
            sensor,
            estimate,
            proposal.world_poses[0],
            np.tile([-2, 2], (7, 1)),
            "approach",
            np.zeros(2),
            admission=receipt,
        )
        == "geometry_profile_mismatch"
    )
    measured = replace(feedback, torque_nm=np.zeros(7), healthy=True)
    assert (
        checks.check(
            sensor,
            estimate,
            proposal.world_poses[0],
            np.tile([-2, 2], (7, 1)),
            "push",
            np.zeros(2),
            profile=OPERATIONAL_V1,
            feedback=measured,
            admission=receipt,
        )
        == "load_control_unverified"
    )
    io.safety = SimpleNamespace(
        provider=SimpleNamespace(last_estimate=estimate), before_command=lambda stage: reason
    )
    with pytest.raises(RuntimeError, match="force_feedback_unavailable"):
        io.execute(Command("delta", proposal.action, "push", receipt))
    io.profile = "legacy-full-state"
    with pytest.raises(ValueError, match="operational IO profile"):
        io.execute(Command("delta", proposal.action, "approach", receipt))


class NumericalIO:
    robot_asset = TEST_ROBOT_REF
    dt = 0.1
    max_ticks = 20

    def __init__(self, binding):
        self.binding = binding
        self.fail_after = None
        self.reset()

    def reset(self):
        self.clock = 0
        self.commands = []
        self.held = 0
        self.stops = 0
        self.pose = ObjectFrame(np.zeros(3), np.eye(3))
        self.neck = []

    def observe(self):
        return sample(self.clock * self.dt, self.clock)

    def tool_pose(self):
        return self.pose

    def set_neck_target(self, value):
        self.neck.append(value)

    def hold_step(self):
        self.held += 1
        self.clock += 1

    def execute(self, command):
        self.commands.append(command)
        if command.kind == "pose":
            self.pose = command.value
        elif command.kind == "delta":
            self.pose = apply_pose_delta(self.pose, command.value)
        self.clock += 1

    def safety_reason(self, stage):
        return "force_limit" if self.fail_after == len(self.commands) else ""

    def stop(self):
        self.stops += 1


@pytest.fixture
def setup_runner(tmp_path, b1_binding):
    release = b1_binding.to_dict()
    release["config"]["inspection"]["waypoints"] = [[0, 0, 0], [0.1, 0.02, 0]]
    release["config"]["inspection"]["sample_times_s"] = [0.1]
    release["config"]["warmup_s"] = 0.1
    binding = type(b1_binding).from_dict(release)
    root = export_dataset(
        [make_b1_episode(binding)], tmp_path / "data", binding, TEST_ROBOT_REF, {"left": "train"}
    )

    def make(family, space, stages=STAGES):
        data = load_b1_data(root, space)
        calls = []

        def predict(features):
            calls.append(features.copy())
            if space == A4_OBJ_CENTRIC_CHUNK:
                stage = stages[min(len(calls) - 1, len(stages) - 1)]
                pose = ObjectFrame(np.array([0.01, 0.02, 0.03]), np.eye(3))
                row = Segment(stage, pose, 0.02, 2, stage == "release").encode()
            else:
                row = np.full(ACTION_DIMS[space], 0.001)
            return np.tile(row, (3, 1))

        model_fixture = SimpleNamespace(
            stats=data.stats, chunk_size=3, predict=predict, seed=lambda seed: calls.clear()
        )
        policy = B1Policy(model_fixture, family, data.dataset.contract, robot_asset=TEST_ROBOT_REF)
        estimator = fake_estimator(binding)
        observer = B1Observer(estimator, binding)
        io = NumericalIO(binding)
        return B1Runner(policy, observer, io), io, estimator, calls

    return make


@pytest.mark.parametrize("family", ["act", "diffusion"])
@pytest.mark.parametrize("space", ACTION_DIMS)
def test_eight_paths_dispatch_and_finish_without_expert_motion(setup_runner, family, space):
    runner, io, _, calls = setup_runner(family, space)
    result = runner.run(max_ticks=12)
    assert io.held == 2 and len(io.neck) == 2  # common prefix, not learned actions
    assert io.stops == 1 and runner.stopped and not runner.pending
    assert runner.source is None
    if space == A4_OBJ_CENTRIC_CHUNK:
        assert result.predicted_complete and result.control_steps == 10
        assert len(calls) == 5  # exactly at segment boundaries
        assert [c.stage for c in io.commands] == [s for s in STAGES for _ in range(2)]
        assert all(c.kind == "pose" for c in io.commands)
    else:
        assert result.reason == "time_budget" and not result.predicted_complete
        assert len(calls) == 4
        assert all(
            c.kind == ("joints" if space == A1_JOINT_DELTA else "delta") for c in io.commands
        )


def test_perception_loss_clears_chunk_before_another_command_and_resets_episode(setup_runner):
    runner, io, estimator, _ = setup_runner("act", A2_EE_DELTA)
    original = estimator.update

    def lose_after_first(obs):
        estimator.reason = "low_confidence" if len(io.commands) == 1 else "observed"
        return original(obs)

    estimator.update = lose_after_first
    result = runner.run(max_ticks=8)
    assert result.reason == "low_confidence" and len(io.commands) == 1
    assert not runner.pending and runner.observer.last is None
    estimator.update = original
    estimator.reason = "observed"
    assert runner.run(max_ticks=2).control_steps == 2


def test_physics_stop_never_invents_release_or_an_extra_tick(setup_runner):
    runner, io, _, _ = setup_runner("diffusion", A4_OBJ_CENTRIC_CHUNK)
    io.fail_after = 3
    result = runner.run()
    assert result.reason == "force_limit" and result.control_steps == 3
    assert all(c.stage != "release" for c in io.commands)
    assert io.clock == io.held + 3


def test_missing_a4_stage_and_oversized_duration_abort(setup_runner):
    runner, io, _, _ = setup_runner(
        "act", A4_OBJ_CENTRIC_CHUNK, stages=("approach", "push", "release")
    )
    result = runner.run()
    assert "skip or regress" in result.reason and len(io.commands) == 2
    runner, io, _, _ = setup_runner("act", A4_OBJ_CENTRIC_CHUNK)
    result = runner.run(max_ticks=1)
    assert "duration" in result.reason and not io.commands


def test_timestamp_error_invalidates_before_next_command(setup_runner):
    runner, io, _, _ = setup_runner("act", A2_EE_DELTA)
    observe = io.observe

    def stale():
        obs = observe()
        if len(io.commands):
            obs["time_s"] = 0
        return obs

    io.observe = stale
    result = runner.run()
    assert result.reason == "nonmonotonic_observation" and len(io.commands) == 1


def test_common_substep_safety_diagnostics():
    def snapshot(force=1, gap=0, forbidden=False):
        return dict(
            components="normal_only",
            forbidden=forbidden,
            contacts=[dict(category="positive", force_n=force, separation_m=gap)],
        )

    assert contact_status([], 80)[0] == "missing_contact_diagnostics"
    assert contact_status([snapshot(1), snapshot(81)], 80)[0] == "force_limit"
    assert contact_status([snapshot(forbidden=True)], 80)[0] == "forbidden_contact"
    assert contact_status([snapshot(np.nan)], 80)[0] == "invalid_contact_diagnostics"
    reason, force, gap = contact_status([snapshot(1), snapshot(3, 0.0001)], 80)
    assert reason == "" and force == 2 and gap == 0.0001


def test_purdue_dispatch_reconstructs_absolute_teacher_goal_without_losing_rotation():
    from scipy.spatial.transform import Rotation

    called = []
    io = object.__new__(PurdueIO)
    io.safety = SimpleNamespace()
    io.env = SimpleNamespace(
        device="cpu",  # numerical command-routing fixture, not a simulator
        command_pose=lambda p, q: called.append((p, q)),
        step=lambda placeholder: called.append(placeholder.numpy()),
        step_a1=lambda value: called.append(value.copy()),
    )
    start = ObjectFrame(np.array([0.1, 0.2, 0.3]), Rotation.from_rotvec([0.2, 0.1, 0]).as_matrix())
    io.tool_pose = lambda: start
    delta = np.array([0.02, -0.03, 0.04, 0.12, -0.1, 0.02])
    goal = apply_pose_delta(start, delta)
    io.execute(Command("delta", delta))
    reconstructed = ObjectFrame(called[0][0], Rotation.from_quat(called[0][1]).as_matrix())
    np.testing.assert_allclose(pose_delta(reconstructed, goal), 0, atol=1e-14)
    np.testing.assert_array_equal(called[1], np.zeros((1, 6)))
    called.clear()
    io.execute(Command("joints", np.arange(7, dtype=float)))
    assert len(called) == 1  # joint route must not call the pose/IK interface
    np.testing.assert_array_equal(called[0], np.arange(7))


def test_matched_replay_uses_recorded_boundaries_but_never_recorded_geometry(b1_binding):
    record = SimpleNamespace(actions=np.arange(21).reshape(3, 7), times=np.array([5, 5.2, 5.5]))
    dataset = SimpleNamespace(
        by_id=lambda _: record,
        binding=b1_binding,
        robot_asset=TEST_ROBOT_REF,
        action_space=A1_JOINT_DELTA,
    )
    replay = MatchedReplay(dataset, "episode")
    source = replay.chunk_source(lambda observation: observation.features)
    for i, time in enumerate([10, 10.2, 10.5]):
        np.testing.assert_array_equal(
            source(SimpleNamespace(time_s=time)), record.actions[i : i + 1]
        )
    with pytest.raises(ValueError, match="exhausted"):
        source(SimpleNamespace(time_s=10.8))
    replay.reset()
    source(SimpleNamespace(time_s=1))
    with pytest.raises(ValueError, match="timing"):
        source(SimpleNamespace(time_s=1.3))
