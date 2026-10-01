"""One causal observed-only encoding path for B1 offline preparation and rollout."""

from dataclasses import dataclass

import numpy as np

from alexdoor_xas.action.b1 import finite_vector
from alexdoor_xas.perception.contracts import (
    LEGACY_FULL_STATE,
    OPERATIONAL_V1,
    DoorEstimate,
    geometric_admission,
    geometry_profile,
    validate_contact_transition,
    validate_geometry,
)
from alexdoor_xas.policies.common.b1_contract import OBS_KEYS
from alexdoor_xas.recording.b1 import OBS_KEYS as SENSOR_KEYS


def numpy(value):
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    return np.asarray(value)


@dataclass(frozen=True)
class PolicyObservation:
    time_s: float
    frame: int
    features: np.ndarray | None
    estimate: DoorEstimate
    reason: str
    geometry_profile: str = LEGACY_FULL_STATE
    generation: int | None = None
    max_age_s: float = 0.15

    @property
    def features_available(self):
        return (
            self.features is not None
            and np.asarray(self.features).ndim == 1
            and np.isfinite(self.features).all()
        )

    @property
    def valid(self):
        if self.geometry_profile == LEGACY_FULL_STATE:
            return self.features is not None and self.estimate.valid
        return self.features_available and self.generation is not None and geometric_admission(
            self.estimate, self.time_s, self.max_age_s,
            profile=self.geometry_profile, generation=self.generation
        ).qualified


def require_estimate(estimate, now, max_age, *, profile=LEGACY_FULL_STATE, generation=None):
    validate_geometry(estimate, now, max_age, profile=profile, generation=generation)


def observation_columns(vector, binding):
    vector = finite_vector(vector, binding.obs_dim)
    static, recent = binding.config["visual_dims"]
    visual = static + recent
    return dict(
        zip(
            OBS_KEYS,
            (
                vector[:static],
                vector[static:visual],
                vector[visual : visual + 9],
                vector[visual + 9 :],
            ),
            strict=True,
        )
    )


class B1Observer:
    """Assemble a provider's two visual blocks and current proprioception.

    The provider exposes binding, NumPy encoding, reset() and update(sensor)->DoorEstimate.
    It owns device conversion and inference, and verifies release artifacts before
    loading. No qualified provider is supplied by this package yet.
    """

    def __init__(self, provider, binding):
        if provider.binding != binding:
            raise ValueError("Perception provider does not match the B1 release")
        self.provider, self.binding = provider, binding
        self.profile = geometry_profile(binding)
        if self.profile == OPERATIONAL_V1 and not hasattr(provider, "generation"):
            raise ValueError("Operational provider must declare its episode generation")
        self.reset()

    def reset(self):
        self.provider.reset()
        self.last = None
        self.last_time = None
        self.last_frame = None
        self.last_contact = None
        self.last_generation = getattr(self.provider, "generation", None)

    def invalidate(self, time, frame, reason):
        self.last = None
        return PolicyObservation(
            time, frame, None, DoorEstimate(time, False, reason), reason, self.profile,
            getattr(self.provider, "generation", None), self.binding.config["max_gap_s"]
        )

    def operational_observation(self, estimate, t, frame, q, dq):
        generation = self.provider.generation
        if generation != self.last_generation:
            self.last = self.last_contact = None
            self.last_generation = generation
        if estimate.reason == "between_inference_ticks" and self.last is not None:
            estimate, encoding = self.last
        elif self.provider.encoding is not None and estimate.operational is not None:
            encoding = finite_vector(self.provider.encoding, self.binding.obs_dim - 18).copy()
            self.last = estimate, encoding
        else:
            return self.invalidate(t, frame, estimate.reason)
        state = estimate.operational
        if state.generation != generation:
            return self.invalidate(t, frame, "wrong_episode_generation")
        if state.contact is not None:
            try:
                validate_contact_transition(self.last_contact, state.contact)
            except ValueError as error:
                return self.invalidate(t, frame, str(error))
            self.last_contact = state.contact
        geometry = geometric_admission(
            estimate, t, self.binding.config["max_gap_s"],
            profile=self.profile, generation=generation
        )
        return PolicyObservation(t, frame, np.r_[encoding, q, dq], estimate,
                                 geometry.reason, self.profile, generation,
                                 self.binding.config["max_gap_s"])

    def update(self, observation):
        # Selection happens before inference: metadata/labels cannot enter the estimator.
        sensor = {key: observation[key] for key in SENSOR_KEYS}
        t, frame = float(sensor["time_s"]), int(sensor["frame"])
        if (
            not np.isfinite(t)
            or self.last_time is not None
            and (t <= self.last_time or frame <= self.last_frame)
        ):
            self.reset()
            return self.invalidate(t, frame, "nonmonotonic_observation")
        self.last_time, self.last_frame = t, frame
        try:
            q = finite_vector(numpy(sensor["joint_position"]), 9)
            dq = finite_vector(numpy(sensor["joint_velocity"]), 9)
        except ValueError:
            self.provider.reset()
            self.last_contact = None
            return self.invalidate(t, frame, "invalid_proprioception")
        estimate = self.provider.update(sensor)
        if self.profile == OPERATIONAL_V1:
            return self.operational_observation(estimate, t, frame, q, dq)
        if estimate.valid:
            encoding = finite_vector(self.provider.encoding, self.binding.obs_dim - 18)
            self.last = (estimate, encoding.copy())
        elif estimate.reason != "between_inference_ticks":
            return self.invalidate(t, frame, estimate.reason)
        if self.last is None:
            return self.invalidate(t, frame, estimate.reason)
        estimate, encoding = self.last
        try:
            require_estimate(estimate, t, self.binding.config["max_gap_s"])
        except ValueError:
            return self.invalidate(t, frame, "stale_or_invalid_estimate")
        features = np.r_[encoding, q, dq]
        return PolicyObservation(t, frame, features, estimate, "observed")
