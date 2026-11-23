"""One causal observed-only encoding path for B1 offline preparation and rollout."""

from dataclasses import dataclass, replace

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
from alexdoor_xas.policies.common.b1_contract import OBS_KEYS, PilotBinding
from alexdoor_xas.recording.b1 import OBS_KEYS as SENSOR_KEYS
from alexdoor_xas.recording.b1 import as_numpy as numpy


@dataclass(frozen=True)
class FeatureBlock:
    values: np.ndarray
    acquired_s: float
    available_s: float
    frame: int
    generation: int

    def require(self, now, frame, generation, width, *, max_age=None):
        finite_vector(self.values, width)
        if (
            type(self.generation) is not int
            or self.generation < 0
            or self.generation != generation
            or not np.isfinite([self.acquired_s, self.available_s, now]).all()
            or not 0 <= self.acquired_s <= self.available_s <= now
            or not isinstance(self.frame, (int, np.integer))
            or not 0 <= self.frame <= frame
        ):
            raise ValueError("unavailable_feature_support")
        if max_age is not None and now - self.acquired_s > max_age + 1e-9:
            raise ValueError("stale_policy_features")


@dataclass(frozen=True)
class VisualEncoding:
    static: FeatureBlock
    recent: FeatureBlock

    def require(self, now, frame, generation, binding):
        static, recent = binding.config["visual_dims"]
        self.static.require(now, frame, generation, static)
        self.recent.require(now, frame, generation, recent, max_age=binding.config["max_gap_s"])

    @property
    def vector(self):
        return np.r_[self.static.values, self.recent.values]


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
    feature_support: VisualEncoding | None = None

    @property
    def features_available(self):
        if (
            self.features is None
            or np.asarray(self.features).ndim != 1
            or not np.isfinite(self.features).all()
        ):
            return False
        if self.feature_support is not None:
            try:
                for block, age in (
                    (self.feature_support.static, None),
                    (self.feature_support.recent, self.max_age_s),
                ):
                    block.require(
                        self.time_s, self.frame, self.generation, np.size(block.values), max_age=age
                    )
                if not np.array_equal(self.features[:-18], self.feature_support.vector):
                    return False
            except (ValueError, TypeError, AttributeError):
                return False
        return True

    @property
    def geometry_available(self):
        return geometric_admission(
            self.estimate,
            self.time_s,
            self.max_age_s,
            profile=self.geometry_profile,
            generation=self.generation,
        ).qualified

    @property
    def valid(self):
        if self.geometry_profile == LEGACY_FULL_STATE:
            return self.features is not None and self.estimate.valid
        return self.features_available and self.generation is not None and self.geometry_available


def require_features(observation, binding):
    """Policy availability is independent of geometry and command admission."""
    finite_vector(observation.features, binding.obs_dim)
    if isinstance(binding, PilotBinding):
        support = observation.feature_support
        if support is None or observation.generation is None:
            raise ValueError("missing_policy_feature_support")
        support.require(observation.time_s, observation.frame, observation.generation, binding)
        if not np.array_equal(observation.features[:-18], support.vector):
            raise ValueError("policy_feature_support_mismatch")


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


def require_provider_binding(provider, binding):
    """Share the pilot geometry/visual recipe contract with the execution IO."""
    if provider is None or provider.binding == binding:
        return
    if not isinstance(binding, PilotBinding):
        raise ValueError("Perception provider does not match the B1 release")
    config = provider.binding.config
    if config.get("inspection") != binding.config["inspection"] or not (
        0 < config.get("max_gap_s", float("inf")) <= binding.config["max_gap_s"]
    ):
        raise ValueError("Pilot geometry inspection or freshness mismatch")


class B1Observer:
    """Assemble visual blocks and current proprioception on one causal clock.

    Releases retain provider-owned encoding and artifact verification. Pilots use
    an independent encoder and an optional geometry provider. No qualified
    provider or trained encoder is supplied by this package yet.
    """

    def __init__(self, provider, binding, *, encoder=None):
        self.pilot = isinstance(binding, PilotBinding)
        require_provider_binding(provider, binding)
        if self.pilot:
            if encoder is None or encoder.binding != binding:
                raise ValueError("Pilot requires an independently bound policy encoder")
        elif encoder is not None or provider is None:
            raise ValueError("Release observations require their bound perception provider")
        self.encoder = encoder
        self.generation = -1
        self.provider, self.binding = provider, binding
        self.profile = geometry_profile(binding)
        if (
            provider is not None
            and self.profile == OPERATIONAL_V1
            and not hasattr(provider, "generation")
        ):
            raise ValueError("Operational provider must declare its episode generation")
        self.reset()

    def reset(self):
        if self.provider is not None:
            self.provider.reset()
        self.generation += 1
        if self.encoder is not None:
            self.encoder.reset()
        self.last_features = None
        self.last_geometry = None
        self.last = None
        self.last_time = None
        self.last_frame = None
        self.last_contact = None
        self.last_generation = getattr(self.provider, "generation", None)

    def invalidate(self, time, frame, reason):
        self.last = None
        return PolicyObservation(
            time,
            frame,
            None,
            DoorEstimate(time, False, reason),
            reason,
            self.profile,
            getattr(self.provider, "generation", None),
            self.binding.config["max_gap_s"],
        )

    def policy_observation(self, sensor, t, frame, q, dq):
        generation = getattr(self.provider, "generation", self.generation)
        if generation != self.last_generation:
            self.last_features = self.last_geometry = None
            self.last_generation = generation
        # Encoder never receives calibration, geometry, teacher labels or feedback.
        inputs = {k: sensor[k] for k in SENSOR_KEYS if k not in ("camera_world", "intrinsics")}
        rgb, depth, mask = (numpy(inputs[k]) for k in ("rgb", "depth_m", "valid_depth"))
        if (
            depth.ndim != 3
            or depth.shape[-1] != 1
            or mask.shape != depth.shape
            or rgb.shape != (*depth.shape[:2], 3)
            or mask.dtype.kind != "b"
            or np.any(mask & (~np.isfinite(depth) | (depth <= 0)))
            or not np.any(mask)
        ):
            self.last_features = None
            return self.invalidate(t, frame, "invalid_rgbd")
        estimate = (
            self.provider.update(sensor)
            if self.provider is not None
            else DoorEstimate(t, False, "geometry_unavailable")
        )
        current_generation = getattr(self.provider, "generation", self.generation)
        if current_generation != generation:
            self.last_features = self.last_geometry = None
            generation = self.last_generation = current_generation
        if estimate.reason == "between_inference_ticks" and self.last_geometry is not None:
            estimate = self.last_geometry
        else:
            self.last_geometry = estimate
        encoded = self.encoder.update(inputs, generation=generation)
        if encoded is not None:
            self.last_features = encoded
        try:
            if self.last_features is None:
                raise ValueError("policy_features_unavailable")
            self.last_features.require(t, frame, generation, self.binding)
        except ValueError as error:
            self.last_features = None
            return replace(
                self.invalidate(t, frame, str(error)), estimate=estimate, generation=generation
            )
        geometry = geometric_admission(
            estimate,
            t,
            self.binding.config["max_gap_s"],
            profile=self.profile,
            generation=generation,
        )
        return PolicyObservation(
            t,
            frame,
            np.r_[self.last_features.vector, q, dq],
            estimate,
            geometry.reason,
            self.profile,
            generation,
            self.binding.config["max_gap_s"],
            self.last_features,
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
            estimate,
            t,
            self.binding.config["max_gap_s"],
            profile=self.profile,
            generation=generation,
        )
        return PolicyObservation(
            t,
            frame,
            np.r_[encoding, q, dq],
            estimate,
            geometry.reason,
            self.profile,
            generation,
            self.binding.config["max_gap_s"],
        )

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
            if self.pilot:
                self.reset()
            else:
                self.provider.reset()
            self.last_contact = None
            return self.invalidate(t, frame, "invalid_proprioception")
        if self.pilot:
            return self.policy_observation(sensor, t, frame, q, dq)
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
