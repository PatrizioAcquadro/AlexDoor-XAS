"""Opt-in temporal ACT bridge to the maintained B1 A1 runner.

Producer ingress supplies real synchronized events, including acknowledged
applied targets. Geometry/admission/controller/stop logic remains in B1Runner.
"""

import numpy as np

from alexdoor_xas.action.spaces import A1_JOINT_DELTA
from alexdoor_xas.perception.contracts import OPERATIONAL_V1, geometry_profile
from alexdoor_xas.policies.act.temporal import TemporalActPolicy
from alexdoor_xas.policies.temporal import IncrementalObservationBuilder


class TemporalA1Execution:
    def __init__(self, policy, binding, robot_asset, observations, *, reuse=False):
        cfg = observations.config
        if (
            policy.config != cfg
            or cfg.history_s != 4
            or policy.chunk_size != 12
            or policy.n_action_steps != 6
            or geometry_profile(binding) != OPERATIONAL_V1
            or not np.isclose(cfg.static_time_s, binding.config["inspection"]["waypoints"][-1][0])
            or not np.isclose(cfg.warmup_s, binding.config["warmup_s"])
        ):
            raise ValueError("Temporal execution requires the explicit 4s/12/6 A1 inspection recipe")
        self.policy, self.binding, self.robot_asset = policy, binding, robot_asset
        self.observations, self.reuse = observations, bool(reuse)
        self.action_space, self.chunk_size = A1_JOINT_DELTA, 12
        self.snapshot = None

    def reset(self, seed=0):
        self.snapshot = None
        self.policy.reset(self.observations.episode_id, self.observations.generation)

    def validate_observation(self, observation):
        # Called on EVERY executed row, including rows pending from a chunk.
        # No prediction from an older query bypasses current sensory freshness.
        self.snapshot = None
        if observation.generation != self.observations.generation:
            raise ValueError("Temporal execution generation mismatch")
        snapshot = self.observations.build(observation.time_s)
        self.policy.validate_snapshot(snapshot, now_s=observation.time_s)
        self.snapshot = snapshot

    def chunk_source(self, observe, *, temporal_ensemble=False, n_action_steps=None):
        if temporal_ensemble or n_action_steps not in (None, 6):
            raise ValueError("Temporal ACT executes six ordered rows without ensembling")

        def source(observation):
            if self.snapshot is None or self.snapshot["decision_time_s"] != observation.time_s:
                raise ValueError("Validate current temporal observations before replanning")
            chunk = np.asarray(self.policy.predict(self.snapshot, now_s=observation.time_s,
                                                   reuse=self.reuse))
            if chunk.shape != (12, 7) or not np.isfinite(chunk).all():
                raise ValueError("Invalid temporal A1 action chunk")
            return chunk[:6].copy()

        return source


class TemporalIngressObserver:
    """Keep the existing geometry observer; add explicit native producer events.

    `events(sensor, observation)` drains newly available event dictionaries for
    IncrementalObservationBuilder.append. It must never manufacture applications
    from requested actions. Images use the shared preprocess_rgbd function.
    """

    def __init__(self, observer, execution, events):
        self.observer, self.execution, self.events = observer, execution, events

    def __getattr__(self, name):
        return getattr(self.observer, name)

    def reset(self):
        self.observer.reset()
        self.execution.observations.reset(self.observer.generation)
        self.execution.reset()

    def update(self, sensor):
        observation = self.observer.update(sensor)
        for event in self.events(sensor, observation):
            self.execution.observations.append(**event)
        return observation


def temporal_a1_runner(
    checkpoint, observer, io, *, config, episode_id, events, feedback_contract=None,
    reuse=False, device="cuda:0", admission_source=None,
):
    """Load a compatible checkpoint and explicitly opt into temporal A1 execution.

    Caller owns the native event source and its reset, actual application acks,
    and synchronized clocks. This entry point never launches a simulator.
    """
    from alexdoor_xas.policies.rollout import B1Runner

    if not np.isclose(io.dt, config.control_dt_s, atol=1e-9, rtol=0):
        raise ValueError("Temporal checkpoint/control interval mismatch")
    policy = TemporalActPolicy.from_checkpoint(
        checkpoint, config=config, runtime_asset=io.robot_asset, device=device
    )
    observations = IncrementalObservationBuilder(
        config, episode_id=episode_id, feedback_contract=feedback_contract
    )
    execution = TemporalA1Execution(policy, observer.binding, io.robot_asset, observations, reuse=reuse)
    return B1Runner(
        execution, TemporalIngressObserver(observer, execution, events), io,
        n_action_steps=6, admission_source=admission_source,
    )
