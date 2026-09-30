"""One causal observed-only encoding path for B1 offline preparation and rollout."""

from dataclasses import dataclass

import numpy as np

from alexdoor_xas.action.b1 import checked_pose, finite_vector, panel_pose
from alexdoor_xas.perception.contracts import DoorEstimate
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

    @property
    def valid(self):
        return self.features is not None and self.estimate.valid


def require_estimate(estimate, now, max_age):
    if not estimate.fresh(now, max_age):
        raise ValueError(f"Unavailable/stale observed geometry: {estimate.reason}")
    checked_pose(estimate.frame)
    panel_pose(estimate.frame, estimate.signed_angle)


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

    The provider exposes binding, encoding, reset() and update(sensor)->DoorEstimate.
    It owns device conversion and inference, and verifies release artifacts before
    loading. No qualified provider is supplied by this package yet.
    """

    def __init__(self, provider, binding):
        if provider.binding != binding:
            raise ValueError("Perception provider does not match the B1 release")
        self.provider, self.binding = provider, binding
        self.reset()

    def reset(self):
        self.provider.reset()
        self.last = None
        self.last_time = None
        self.last_frame = None

    def invalidate(self, time, frame, reason):
        self.last = None
        return PolicyObservation(time, frame, None, DoorEstimate(time, False, reason), reason)

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
            return self.invalidate(t, frame, "invalid_proprioception")
        estimate = self.provider.update(sensor)
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
