"""Operational contact evidence; never a measurement of evaluator contact gaps."""

from collections import deque
from dataclasses import dataclass

import numpy as np

from alexdoor_xas.action.b1 import checked_pose
from alexdoor_xas.perception.admission import require_current_admission, rotational_travel
from alexdoor_xas.perception.contracts import validate_local_contact


@dataclass(frozen=True)
class ContactMonitorConfig:
    """Measured pilot recipe, separate from physical evaluator thresholds.

    normal_model_bound_m includes visible-surface/collision-model disagreement;
    patch support and calibrated robot errors are added independently. Neither
    this bound nor a plane-fit residual establishes directly observed contact.
    Optional robot_fk bounds describe measured tool kinematics, independently of
    command-tracking error. Seek travel is cumulative measured world tool travel.
    """

    normal_model_bound_m: float
    force_on_n: float
    force_off_n: float
    confirmation_s: float
    max_sample_gap_s: float
    max_seek_s: float
    max_seek_travel_m: float
    force_window_s: float
    robot_fk_position_bound_m: float | None = None
    robot_fk_rotation_bound_rad: float | None = None

    def __post_init__(self):
        values = [v for v in vars(self).values() if v is not None]
        if (self.robot_fk_position_bound_m is None) != (self.robot_fk_rotation_bound_rad is None):
            raise ValueError("partial_robot_fk_contact_bound")
        if not np.isfinite(values).all() or min(values) < 0:
            raise ValueError("invalid_contact_monitor_recipe")
        if not (
            self.force_on_n > self.force_off_n >= 0
            and 0 < self.confirmation_s < self.max_seek_s
            and 0 < self.max_sample_gap_s <= self.confirmation_s
            and self.max_seek_travel_m > 0
            and self.max_sample_gap_s <= self.force_window_s < self.max_seek_s
        ):
            raise ValueError("invalid_contact_monitor_recipe")


@dataclass(frozen=True)
class ContactEvidence:
    status: str
    reason: str
    time_s: float
    finger_separation_intervals_m: tuple = ()
    normal_force_lower_n: float | None = None
    uncertainty_m: float | None = None
    directly_observed: bool = False


class ObservedContactMonitor:
    """Fuse qualified local identity/support, robot geometry/load and continuity.

    A valid receipt is required; this monitor cannot replace swept-space,
    footprint, response or stopping admission. Occluded fingertips can have
    indirect contact evidence, but never a fabricated depth measurement.
    Loss/contradiction after contact latches until explicit episode reset.
    Withdrawal is recognized only from the submitted robot motion, never added.
    """

    def __init__(self, config, distal_faces):
        self.config = config
        self.faces = tuple(np.array(face, dtype=float, copy=True) for face in distal_faces)
        if len(self.faces) != 2 or any(
            face.ndim != 2 or face.shape[1:] != (3,) or not len(face) or not np.isfinite(face).all()
            for face in self.faces
        ):
            raise ValueError("missing_calibrated_two_finger_geometry")
        self.reset()

    def reset(self):
        self.identity = self.last = self.result = None
        self.seek_started = self.seek_anchor = self.confirm_started = None
        self.seek_travel_m = 0.0
        self.established = self.releasing = False
        self.latched_reason = None
        self.force_samples = deque()

    def contact_load(self, load, now):
        samples = load.normal_samples
        if not samples:
            return None
        for start, end, value in samples:
            if (
                not np.isfinite([start, end, value]).all()
                or not 0 <= start < end <= load.feedback.timestamp_s
                or now - start > self.config.max_sample_gap_s + 1e-9
            ):
                raise ValueError("invalid_native_contact_load_samples")
            existing = next(
                (
                    v
                    for a, b, v in self.force_samples
                    if abs(a - start) < 1e-9 and abs(b - end) < 1e-9
                ),
                None,
            )
            if existing is not None:
                if not np.isclose(value, existing, atol=1e-9, rtol=0):
                    raise ValueError("different_contact_load_at_same_time")
            else:
                if self.force_samples and abs(start - self.force_samples[-1][1]) > 1e-9:
                    raise ValueError("contact_load_coverage_gap")
                self.force_samples.append((start, end, value))
        end = load.feedback.timestamp_s
        if abs(self.force_samples[-1][1] - end) > 1e-9:
            raise ValueError("invalid_native_contact_load_samples")
        start = end - self.config.force_window_s
        while self.force_samples and self.force_samples[0][1] <= start:
            self.force_samples.popleft()
        if not self.force_samples or self.force_samples[0][0] > start + 1e-9:
            return None
        # Explicit covered physics intervals; do not interpolate across missing
        # measurements or turn duplicate before/after checks into extra evidence.
        return float(
            sum((b - max(a, start)) * v for a, b, v in self.force_samples)
            / self.config.force_window_s
        )

    def update(self, now, estimate, tool, load, admission):
        if self.latched_reason:
            return ContactEvidence("lost", self.latched_reason, now)
        try:
            checked_pose(tool)
            require_current_admission(admission, estimate, now)
            patch = validate_local_contact(estimate.local, now)
            if admission.local_evidence is None:
                raise ValueError("missing_local_contact_admission")
            if patch.tracking_state != "tracking" or patch.ownership not in (
                "observed_rigid_material",
                "observed_moving_rigid_candidate",
            ):
                raise ValueError("unresolved_contact_material")
            if load.model_verified is not True or load.stop_verified is not True:
                raise ValueError("unverified_contact_load_model")
            if (
                load.feedback.healthy is not True
                or load.feedback.torque_nm is None
                or not load.feedback.timestamp_s <= load.feedback.available_s <= now
            ):
                raise ValueError("unavailable_contact_feedback")
            if not 0 <= now - load.feedback.timestamp_s <= self.config.max_sample_gap_s + 1e-9:
                raise ValueError("stale_contact_feedback")
            if not np.isfinite([load.normal_lower_n, load.force_upper_n]).all():
                raise ValueError("invalid_contact_load")
            identity = (
                estimate.local.generation,
                patch.patch_id,
                estimate.local.selection.selection_id,
            )
            if self.identity is not None and identity != self.identity:
                raise ValueError("contact_material_changed")
            if self.last is not None and (
                now < self.last[0] or now - self.last[0] > self.config.max_sample_gap_s + 1e-9
            ):
                raise ValueError("contact_observation_gap")
            self.identity = identity
            normal = patch.world_pose.rot[:, 0]
            points = [face @ tool.rot.T + tool.origin for face in self.faces]
            signed = np.array(
                [np.min((patch.world_pose.origin - face) @ normal) for face in points]
            )
            radius = max(
                np.linalg.norm(face - patch.world_pose.origin, axis=1).max() for face in points
            )
            support, limits = patch.pose_support, admission.limits
            motion_uncertainty = (
                self.config.normal_model_bound_m
                + patch.geometry_support.position_bound_m
                + rotational_travel(radius, patch.geometry_support.rotation_bound_rad)
                + support.position_bound_m
                + rotational_travel(radius, support.rotation_bound_rad)
                + limits.robot_position_bound_m
                + rotational_travel(
                    max(np.linalg.norm(f, axis=1).max() for f in self.faces),
                    limits.robot_rotation_bound_rad,
                )
            )
            robot_position = (
                limits.robot_position_bound_m
                if self.config.robot_fk_position_bound_m is None
                else self.config.robot_fk_position_bound_m
            )
            robot_rotation = (
                limits.robot_rotation_bound_rad
                if self.config.robot_fk_rotation_bound_rad is None
                else self.config.robot_fk_rotation_bound_rad
            )
            if (
                patch.normal_position_bound_m is not None
                and patch.normal_rotation_bound_rad is not None
            ):
                uncertainty = (
                    self.config.normal_model_bound_m
                    + patch.normal_position_bound_m
                    + rotational_travel(radius, patch.normal_rotation_bound_rad)
                    + robot_position
                    + rotational_travel(
                        max(np.linalg.norm(f, axis=1).max() for f in self.faces), robot_rotation
                    )
                )
            else:
                uncertainty = motion_uncertainty
            margins = [float(value) for _, value in admission.margins_m]
            # The receipt already charges geometry, pose and robot uncertainty.
            # Only the additional visible/collision-model discrepancy is absent
            # from that receipt. Do not spend the shared bounds a second time.
            if (
                not margins
                or not np.isfinite(margins).all()
                or self.config.normal_model_bound_m >= min(margins)
            ):
                raise ValueError("contact_uncertainty_exceeds_admitted_margin")
            relative = (tool.origin - patch.world_pose.origin) @ patch.world_pose.rot
            if self.last is not None and now > self.last[0]:
                allowed = (
                    motion_uncertainty
                    + self.last[2]
                    + limits.relative_speed_bound_m_s * (now - self.last[0])
                )
                if np.linalg.norm(relative - self.last[1]) > allowed + 1e-9:
                    raise ValueError("inconsistent_contact_motion")
            if (
                self.last is not None
                and now == self.last[0]
                and (
                    not np.allclose(relative, self.last[1], atol=1e-9, rtol=0)
                    or not np.allclose(tool.origin, self.last[3], atol=1e-9, rtol=0)
                )
            ):
                raise ValueError("different_contact_pose_at_same_time")
            # Same-clock before/after checks do not accumulate artificial support.
            intervals = tuple((float(x - uncertainty), float(x + uncertainty)) for x in signed)
            start, end = admission.proposal.world_poses
            # Every calibrated distal point must withdraw. Center translation
            # alone can hide one finger rotating further into the material.
            displacement = np.concatenate(
                [face @ (end.rot - start.rot).T + end.origin - start.origin for face in self.faces]
            )
            direction_error = rotational_travel(
                np.linalg.norm(displacement, axis=1), support.rotation_bound_rad
            )
            withdrawing = bool(np.all(-displacement @ normal > direction_error + 1e-9))
            if self.established and withdrawing:
                self.releasing = True
            elif (
                self.releasing and not withdrawing and load.normal_lower_n >= self.config.force_on_n
            ):
                self.releasing = False
            plausible = bool(np.any(np.abs(signed) <= uncertainty))
            mean = self.contact_load(load, now)
            force = None if mean is None else float(mean)
            positive = force is not None and force >= (
                self.config.force_off_n if self.established else self.config.force_on_n
            )
            if (
                load.normal_lower_n >= self.config.force_on_n
                and not plausible
                and not self.established
            ):
                raise ValueError("load_without_intended_surface")
            if self.established and not positive:
                if self.releasing:
                    status, reason = "separated", "commanded_withdrawal"
                    self.established = False
                    self.confirm_started = None
                else:
                    self.latched_reason = "contact_load_lost_or_uncertain"
                    status, reason = "lost", self.latched_reason
            elif self.established and not plausible:
                self.latched_reason = "contact_geometry_inconsistent"
                status, reason = "lost", self.latched_reason
            elif plausible and positive:
                self.confirm_started = now if self.confirm_started is None else self.confirm_started
                self.established = now - self.confirm_started >= self.config.confirmation_s - 1e-9
                status = "contact" if self.established else "seeking"
                reason = "supported_indirect_contact" if self.established else "confirming_contact"
            elif self.releasing:
                status, reason = "separated", "commanded_withdrawal"
            elif np.min(signed - uncertainty) > 0:
                self.confirm_started = None
                status, reason = "separated", "observed_geometric_separation"
            else:
                self.confirm_started = None
                status, reason = (
                    "seeking",
                    (
                        "insufficient_contact_load_history"
                        if mean is None
                        else "normal_contact_not_established"
                    ),
                )
            if admission.proposal.loaded and not self.releasing and status != "contact":
                if self.seek_started is None:
                    self.seek_started, self.seek_anchor = now, tool.origin.copy()
                else:
                    self.seek_travel_m += float(np.linalg.norm(tool.origin - self.seek_anchor))
                    self.seek_anchor = tool.origin.copy()
                if (
                    now - self.seek_started > self.config.max_seek_s
                    or self.seek_travel_m > self.config.max_seek_travel_m
                ):
                    self.latched_reason = "contact_acquisition_budget"
                    status, reason = "lost", self.latched_reason
            self.last = now, relative.copy(), motion_uncertainty, tool.origin.copy()
            self.result = ContactEvidence(status, reason, now, intervals, force, uncertainty)
            return self.result
        except (ValueError, TypeError, AttributeError, IndexError) as error:
            self.confirm_started = None
            if self.established:
                self.latched_reason = str(error)
            self.result = ContactEvidence("unavailable", str(error), now)
            return self.result
