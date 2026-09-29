"""One causal observed-only encoding path for B1 offline preparation and rollout."""

from dataclasses import dataclass

import numpy as np

from alexdoor_xas.action.b1 import checked_pose, finite_vector, panel_pose
from alexdoor_xas.perception.model import DoorEstimate
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
    h = binding.config["hidden_size"]
    return dict(
        zip(
            OBS_KEYS,
            (vector[:h], vector[h : 2 * h], vector[2 * h : 2 * h + 9], vector[2 * h + 9 :]),
            strict=True,
        )
    )


def load_frozen_observer(release_path, checkpoint_path, backbone_path, *, device="cuda:0"):
    """Bind real artifact bytes only after 6.0 supplies its qualified/frozen release."""
    import hashlib
    import json
    from pathlib import Path

    from alexdoor_xas.perception.model import FrozenBackbone, ObservedEstimator
    from alexdoor_xas.perception.training import load_checkpoint
    from alexdoor_xas.policies.common.b1_contract import PerceptionBinding

    binding = PerceptionBinding.from_dict(json.loads(Path(release_path).read_text()))
    release = binding.to_dict()
    for path, key in (
        (checkpoint_path, "checkpoint_sha256"),
        (Path(backbone_path) / "model.safetensors", "backbone_sha256"),
    ):
        with Path(path).open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        if digest != release[key]:
            raise ValueError(f"Frozen perception artifact mismatch: {key}")
    estimator, _ = load_checkpoint(checkpoint_path, binding.config, device)
    if not estimator.confidence_qualified:
        raise ValueError("Perception checkpoint confidence is unqualified")
    estimator.requires_grad_(False).eval()
    backbone = FrozenBackbone(backbone_path).to(device)
    return B1Observer(
        ObservedEstimator(backbone, estimator, binding.config, policy_encoding=True), binding
    )


class B1Observer:
    def __init__(self, estimator, binding):
        if estimator.config != binding.config or not estimator.policy_encoding:
            raise ValueError("Observed estimator does not match the B1 encoding contract")
        self.estimator, self.binding = estimator, binding
        self.reset()

    def reset(self):
        self.estimator.reset()
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
            self.estimator.reset()
            return self.invalidate(t, frame, "invalid_proprioception")
        estimate = self.estimator.update(sensor)
        if estimate.valid:
            encoding = finite_vector(self.estimator.encoding, self.binding.obs_dim - 18)
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
