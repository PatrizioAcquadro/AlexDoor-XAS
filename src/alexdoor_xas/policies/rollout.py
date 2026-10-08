"""Observed B1 action dispatch and bounded rollout, independent of simulator imports."""

from collections import deque
from dataclasses import dataclass

import numpy as np

from alexdoor_xas.action.b1 import (
    ACTION_DIMS,
    Segment,
    SegmentMotion,
    StageSequence,
    apply_pose_delta,
    finite_vector,
)
from alexdoor_xas.action.frames import frame_delta_to_world
from alexdoor_xas.action.spaces import A1_JOINT_DELTA, A3_OBJ_REL_EE_DELTA, A4_OBJ_CENTRIC_CHUNK
from alexdoor_xas.assets.identity import assert_checkpoint_runtime_compatible
from alexdoor_xas.perception.admission import ActionAdmission, require_current_admission
from alexdoor_xas.perception.contracts import (
    LEGACY_FULL_STATE,
    OPERATIONAL_V1,
    geometry_profile,
    same_pose,
    validate_reference,
)
from alexdoor_xas.policies.observations import require_estimate


@dataclass(frozen=True)
class Command:
    kind: str  # joints, delta, pose
    value: object
    stage: str | None = None
    admission: ActionAdmission | None = None


class ActionAdapter:
    """Transform only model predictions; no expert, surface target or success angle."""

    def __init__(self, space, max_age_s, *, profile=LEGACY_FULL_STATE, source="policy"):
        if space not in ACTION_DIMS:
            raise ValueError("Unknown B1 action space")
        self.space, self.max_age_s = space, max_age_s
        if profile not in (LEGACY_FULL_STATE, OPERATIONAL_V1) or source not in (
            "policy",
            "diagnostic",
        ):
            raise ValueError("Unknown adapter profile/source")
        self.profile, self.source = profile, source
        self.reset()

    def reset(self):
        self.sequence = StageSequence()
        self.motion = None
        self.tick = 0
        self.done = False
        self.admission = None
        self.blocked = False

    def admit(self, observation, tool, action, admission):
        if self.motion is not None:
            if admission is not None and admission is not self.admission:
                raise ValueError("Cannot replace an executing action admission")
            admission = self.admission
        if admission is None:
            raise ValueError("Missing explicit operational action admission")
        require_current_admission(admission, observation.estimate, observation.time_s)
        if observation.generation != admission.estimate.operational.generation:
            raise ValueError("Action admission episode generation mismatch")
        validate_reference(
            observation.estimate,
            observation.time_s,
            self.max_age_s,
            generation=observation.generation,
        )
        proposal = admission.proposal
        if (
            proposal.action_space != self.space
            or proposal.source != self.source
            or (admission.provisional and self.source != "diagnostic")
        ):
            raise ValueError("Action admission source/space mismatch")
        if self.motion is None:
            row = finite_vector(action, ACTION_DIMS[self.space])
            if not np.allclose(row, proposal.action, atol=1e-8, rtol=0):
                raise ValueError("Command differs from admitted action")
            if not same_pose(tool, proposal.world_poses[0]):
                raise ValueError("Tool differs from admitted initial pose")
        self.admission = admission

    def require_goal(self, observation, index, goal=None):
        proposal = self.admission.proposal
        if index >= len(proposal.times_s) or not np.isclose(
            observation.time_s, proposal.times_s[index - 1], atol=1e-8, rtol=0
        ):
            raise ValueError("Command differs from admitted schedule")
        if goal is not None and not same_pose(goal, proposal.world_poses[index]):
            raise ValueError("Command differs from admitted world trajectory")

    def command(self, observation, tool, *, action=None, remaining_ticks, admission=None):
        if self.blocked:
            raise ValueError("Operational adapter stop requires explicit reset")
        try:
            return self._command(
                observation,
                tool,
                action=action,
                remaining_ticks=remaining_ticks,
                admission=admission,
            )
        except (ValueError, TypeError, AttributeError, IndexError):
            if self.profile == OPERATIONAL_V1:
                self.blocked = True
                self.motion = self.admission = None
            raise

    def _command(self, observation, tool, *, action, remaining_ticks, admission):
        if self.done:
            raise ValueError("Action after predicted termination")
        if observation.geometry_profile != self.profile:
            raise ValueError("Observation/adapter geometry profile mismatch")
        if self.profile == LEGACY_FULL_STATE:
            if admission is not None or not observation.valid:
                raise ValueError(f"Invalid observed input: {observation.reason}")
            require_estimate(observation.estimate, observation.time_s, self.max_age_s)
        else:
            if not observation.features_available or (
                self.source == "policy" and not observation.valid
            ):
                raise ValueError(f"Invalid observed input: {observation.reason}")
            self.admit(observation, tool, action, admission)
        if self.space != A4_OBJ_CENTRIC_CHUNK:
            row = finite_vector(action, ACTION_DIMS[self.space])
            if self.profile == OPERATIONAL_V1 and len(self.admission.proposal.world_poses) != 2:
                raise ValueError("Primitive admission requires one control interval")
            if self.space == A1_JOINT_DELTA:
                if self.profile == OPERATIONAL_V1:
                    self.require_goal(observation, 1)
                return Command("joints", row, admission=self.admission)
            if self.space == A3_OBJ_REL_EE_DELTA:
                row = frame_delta_to_world(row, observation.estimate.frame)
            if self.profile == OPERATIONAL_V1:
                self.require_goal(observation, 1, apply_pose_delta(tool, row))
            return Command("delta", row, admission=self.admission)
        if self.motion is None:
            segment = Segment.decode(action, remaining_ticks=remaining_ticks)
            if self.profile == OPERATIONAL_V1 and (
                len(self.admission.proposal.world_poses) != segment.ticks + 1
                or segment.stage in ("contact", "push", "hold")
                and not self.admission.proposal.loaded
            ):
                raise ValueError("A4 admission duration/load mismatch")
            self.sequence.accept(segment)
            reference = (
                self.admission.estimate if self.profile == OPERATIONAL_V1 else observation.estimate
            )
            self.motion = SegmentMotion.start(
                segment, tool, reference.frame, reference.signed_angle
            )
            self.tick = 0
        elif action is not None:
            raise ValueError("Cannot replace an executing A4 segment")
        self.tick += 1
        hinge = (
            self.admission.estimate.frame
            if self.profile == OPERATIONAL_V1
            else observation.estimate.frame
        )
        goal = self.motion.goal(self.tick, hinge)
        if self.profile == OPERATIONAL_V1:
            self.require_goal(observation, self.tick, goal)
        command = Command("pose", goal, self.motion.segment.stage, self.admission)
        if self.tick == self.motion.segment.ticks:
            self.done = self.motion.segment.end_episode
            self.motion = None
        return command


@dataclass(frozen=True)
class RolloutResult:
    reason: str
    control_steps: int
    predicted_complete: bool = False  # not task success or physical qualification


class MatchedReplay:
    """Recorded action source for matched replay diagnostics, never a learned policy.

    A3/A4 are still transformed through the live observer. Recorded geometry is
    not supplied to the adapter. The caller labels results as replay diagnostics.
    """

    def __init__(self, dataset, episode_id):
        self.record = dataset.by_id(episode_id)
        self.binding, self.robot_asset = dataset.binding, dataset.robot_asset
        self.action_space = dataset.action_space
        self.control_ticks = (
            int(self.record.actions[:, 15].sum())
            if self.action_space == A4_OBJ_CENTRIC_CHUNK
            else len(self.record.actions)
        )
        self.reset()

    def reset(self, seed=0):
        self.index = 0
        self.start_time = None

    def chunk_source(self, observe, *, temporal_ensemble=False, n_action_steps=None):
        if temporal_ensemble or n_action_steps is not None:
            raise ValueError("Matched replay consumes exactly one recorded row at a time")

        def source(observation):
            if self.index >= len(self.record.actions):
                raise ValueError("Matched replay exhausted its recorded commands")
            if self.start_time is None:
                self.start_time = observation.time_s
            expected = self.record.times[self.index] - self.record.times[0]
            if not np.isclose(observation.time_s - self.start_time, expected, atol=1e-7, rtol=0):
                raise ValueError("Matched replay action timing differs from the recording")
            row = self.record.actions[self.index].copy()
            self.index += 1
            return row[None]

        return source


class B1Runner:
    """Reset, shared inspection, fresh observations, policy chunks and safety stops.

    IO owns simulator execution and a stop-only safety monitor. The runner sees
    RGB-D/proprioception and robot FK, never door truth or evaluator progress.
    """

    def __init__(self, policy, observer, io, *, temporal_ensemble=False, n_action_steps=None):
        if observer.binding != policy.binding or io.binding != policy.binding:
            raise ValueError("Policy and observer perception bindings differ")
        assert_checkpoint_runtime_compatible(policy.robot_asset, io.robot_asset)
        self.policy, self.observer, self.io = policy, observer, io
        self.source_options = dict(
            temporal_ensemble=temporal_ensemble, n_action_steps=n_action_steps
        )
        self.adapter = ActionAdapter(
            policy.action_space,
            observer.binding.config["max_gap_s"],
            profile=geometry_profile(observer.binding),
        )
        self.pending = deque()
        self.stopped = True

    def reset(self, seed=0):
        self.pending.clear()
        self.adapter.reset()
        self.observer.reset()
        self.policy.reset(seed)
        self.source = self.policy.chunk_source(lambda obs: obs.features, **self.source_options)
        self.io.reset()
        self.stopped = False

    def stop(self, reason, steps, *, complete=False):
        self.pending.clear()
        self.source = None
        self.adapter.reset()
        self.observer.reset()
        self.io.stop()
        self.stopped = True
        return RolloutResult(reason, steps, complete)

    def inspect(self):
        """Use the common neck schedule and hold the parked tool through warmup."""
        cfg = self.observer.binding.config
        points = np.asarray(cfg["inspection"]["waypoints"])
        end = float(points[-1, 0]) + cfg["warmup_s"]
        count = int(np.ceil(end / self.io.dt - 1e-8))
        observation = self.observer.update(self.io.observe())
        for tick in range(count):
            if reason := self.io.safety_reason("inspect"):
                return observation, reason
            time_s = (tick + 1) * self.io.dt
            target = [np.interp(time_s, points[:, 0], points[:, j]) for j in (1, 2)]
            self.io.set_neck_target(target)
            self.io.hold_step()
            observation = self.observer.update(self.io.observe())
        if reason := self.io.safety_reason("inspect"):
            return observation, reason
        return observation, "" if observation.valid else observation.reason

    def run(self, *, max_ticks=None, seed=0):
        default_budget = min(
            self.io.max_ticks, getattr(self.policy, "control_ticks", self.io.max_ticks)
        )
        budget = default_budget if max_ticks is None else max_ticks
        if not isinstance(budget, int) or not 1 <= budget <= self.io.max_ticks:
            raise ValueError("Rollout budget must stay within the shared episode horizon")
        steps = 0
        try:
            self.reset(seed)
            observation, reason = self.inspect()
            if reason:
                return self.stop(reason, steps)
            for _ in range(budget):
                if reason := self.io.safety_reason(None):
                    return self.stop(reason, steps)
                if not observation.valid:
                    return self.stop(observation.reason, steps)
                row = None
                if self.policy.action_space == A4_OBJ_CENTRIC_CHUNK:
                    if self.adapter.motion is None:
                        # Replan at each segment boundary, never average discrete stages.
                        row = self.source(observation)[0]
                else:
                    if not self.pending:
                        self.pending.extend(self.source(observation))
                    row = self.pending.popleft()
                command = self.adapter.command(
                    observation, self.io.tool_pose(), action=row, remaining_ticks=budget - steps
                )
                self.io.execute(command)
                steps += 1
                observation = self.observer.update(self.io.observe())
                if reason := self.io.safety_reason(command.stage):
                    return self.stop(reason, steps)
                if not observation.valid:
                    return self.stop(observation.reason, steps)
                if isinstance(self.policy, MatchedReplay) and steps == self.policy.control_ticks:
                    return self.stop("replay_complete", steps)
                if self.adapter.done:
                    return self.stop("predicted_end", steps, complete=True)
            return self.stop("time_budget", steps)
        except (ValueError, RuntimeError, KeyError, IndexError, TypeError, StopIteration) as error:
            return self.stop(f"invalid_execution: {error}", steps)
        except BaseException:
            self.stop("interrupted_execution", steps)
            raise
