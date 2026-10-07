"""Observed static reference, vertical hinge evidence and independently timed angle.

All bounds are conditional correspondence envelopes, not hardware metrology.
Nothing in this module admits an action or reads evaluator annotations.
"""

from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation

from alexdoor_xas.perception.contracts import FieldSupport
from alexdoor_xas.perception.geometry import normal_frame


@dataclass(frozen=True)
class StaticReference:
    pose: np.ndarray
    rotation: np.ndarray
    support: FieldSupport
    closed_by_protocol: bool = False

    @classmethod
    def observed(cls, pose, support, *, closed_by_protocol=False):
        pose = np.array(pose, dtype=float, copy=True)
        rotation = normal_frame(pose[:3, 0])
        pose.setflags(write=False)
        rotation.setflags(write=False)
        return cls(pose, rotation, support, closed_by_protocol)


@dataclass(frozen=True)
class MotionObservation:
    pose: np.ndarray
    support: FieldSupport
    # Initial-world centroid of the actual measured registration correspondences.
    anchor: np.ndarray


def _rotation(reference, observation):
    delta = observation.pose[:3, :3] @ reference.pose[:3, :3].T
    yaw = float(np.arctan2(delta[1, 0] - delta[0, 1], delta[0, 0] + delta[1, 1]))
    vertical = Rotation.from_rotvec([0, 0, yaw]).as_matrix()
    residual = float(Rotation.from_matrix(vertical.T @ delta).magnitude())
    error = observation.support.rotation_bound_rad + reference.support.rotation_bound_rad
    return yaw, vertical, residual, error


def current_angle(reference, observation):
    """Current accepted image supports angle even when no displacement sample is added."""
    yaw, _, residual, error = _rotation(reference, observation)
    compatible = residual <= error + 1e-12
    return dict(
        relative_angle_rad=yaw if compatible else None,
        rotation_bound_rad=error if compatible else None,
        vertical_residual_rad=residual,
        reason="" if compatible else "current_rotation_incompatible_with_vertical",
        acquired_s=observation.support.acquired_s,
        supported_s=observation.support.supported_s,
        available_s=observation.support.available_s,
        generation=observation.support.generation,
        closed_reference_validated=False,
        closed_by_protocol=reference.closed_by_protocol,
    )


def fit_motion_axis(reference, samples, floor_z, *, floor_bound_m=None, diagnostics=None):
    """Fit a vertical model with >=3 informative samples and >=2/3 consensus.

    Compare distance to the vertical rotation model with measured angular error,
    not the tilt of an individually normalized small rotation axis. All rejected
    observations remain in the caller's history and in the fit diagnostics.
    A positive feedback term propagates rotation error at an uncertain hinge.
    Missing floor/calibration errors do not become zero complete bounds.
    """
    detail = diagnostics if diagnostics is not None else {}
    detail.update(
        reason="insufficient_informative_motion",
        informative_samples=0,
        compatible_samples=0,
        rejected_sample_times_s=[],
    )
    if reference.support.position_bound_m is None or reference.support.rotation_bound_rad is None:
        detail["reason"] = "missing_initial_reference_bound"
        return None
    motions = []
    informative = 0
    initial_inverse = np.linalg.inv(reference.pose)
    for sample in samples:
        if sample.support.generation != reference.support.generation:
            detail["reason"] = "wrong_episode_generation"
            return None
        if sample.support.position_bound_m is None or sample.support.rotation_bound_rad is None:
            continue
        yaw, rotation, residual, angular = _rotation(reference, sample)
        if abs(yaw) <= max(np.deg2rad(5), 2 * angular):
            continue
        informative += 1
        if residual > angular + 1e-12:
            detail["rejected_sample_times_s"].append(sample.support.supported_s)
            continue
        delta = sample.pose @ initial_inverse
        current_anchor = (delta @ np.r_[sample.anchor, 1])[:3]
        a = (np.eye(3) - rotation)[:, :2]
        b = current_anchor - rotation @ sample.anchor
        # Rotational discrepancy from projection is explicit model error, not
        # permission to shrink the observed angular uncertainty.
        k = 2 * np.sin(min(np.pi, angular + residual) / 2)
        base = sample.support.position_bound_m + reference.support.position_bound_m
        motions.append((sample, a, b, base, k))
    detail.update(informative_samples=informative, compatible_samples=len(motions))
    required = max(3, int(np.ceil(2 * informative / 3)))
    if len(motions) < required:
        if informative >= 3:
            detail["reason"] = "insufficient_vertical_consensus"
        return None
    a = np.concatenate([m[1] for m in motions])
    b = np.concatenate([m[2] for m in motions])
    # Fixed uncertainty weights: no evaluator-driven sample deletion or retuning.
    scales = np.array(
        [
            base
            + k
            * (
                np.linalg.norm(sample.anchor - reference.pose[:3, 3])
                + abs(reference.pose[2, 3] - floor_z)
            )
            for sample, _, _, base, k in motions
        ]
    )
    weights = np.repeat(1 / np.maximum(scales, 1e-12), 3)
    weighted = a * weights[:, None]
    singular = np.linalg.svd(weighted, compute_uv=False)
    if singular[-1] <= 0 or singular[0] / singular[-1] > 10:
        detail["reason"] = "ill_conditioned_axis"
        return None
    inverse = np.linalg.pinv(weighted) * weights
    xy = inverse @ b
    origin = np.r_[xy, floor_z]
    residuals = np.linalg.norm((a @ xy - b).reshape(-1, 3), axis=1)
    levers = np.array([np.linalg.norm(origin - m[0].anchor) for m in motions])
    k = np.array([m[4] for m in motions])
    envelopes = np.array([m[3] for m in motions]) + k * levers
    consensus = residuals <= envelopes + 1e-12
    detail.update(
        residuals_m=residuals.tolist(),
        envelopes_m=envelopes.tolist(),
        residual_consensus_samples=int(consensus.sum()),
    )
    # Require a common hinge explanation; conflicting motions are not silently discarded.
    if not consensus.all():
        detail["reason"] = "inconsistent_axis_residuals"
        return None
    sensitivity = np.abs(inverse)
    base_bound = float(np.linalg.norm(sensitivity @ np.repeat(envelopes, 3)))
    feedback = float(np.linalg.norm(sensitivity @ np.repeat(k, 3)))
    conditional = base_bound / (1 - feedback) if feedback < 1 else None
    # Under the vertical model, floor uncertainty affects Z only. Sensor/FK
    # calibration is not supplied by this diagnostic and remains a missing source.
    floor_bound = None if floor_bound_m is None else float(floor_bound_m)
    with_floor = (
        None
        if conditional is None or floor_bound is None
        else float(np.hypot(conditional, floor_bound))
    )
    detail["reason"] = "" if conditional is not None else "unbounded_rotation_feedback"
    support_times = [m[0].support.supported_s for m in motions]
    return dict(
        origin=origin,
        direction=np.array([0.0, 0.0, 1.0]),
        static_rotation=reference.rotation.copy(),
        conditional_position_bound_m=conditional,
        position_bound_m=None,  # complete floor/calibration/FK budget is unmeasured
        conditional_with_floor_bound_m=with_floor,
        floor_bound_m=floor_bound,
        missing_bound_sources=["calibration_fk"] + (["floor"] if floor_bound is None else []),
        bound_before_feedback_m=base_bound,
        bound_feedback=feedback,
        residual_m=float(residuals.max()),
        condition=float(singular[0] / singular[-1]),
        acquired_s=min(reference.support.acquired_s, *(m[0].support.acquired_s for m in motions)),
        supported_s=max(support_times),
        available_s=max(m[0].support.available_s for m in motions),
        generation=reference.support.generation,
        sample_times_s=support_times,
        reason=detail["reason"],
        closed_reference_validated=False,
        closed_by_protocol=reference.closed_by_protocol,
    )
