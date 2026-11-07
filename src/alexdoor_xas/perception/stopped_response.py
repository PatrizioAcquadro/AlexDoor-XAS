"""Stopped-only compensation from the last completed calibrated robot residual."""

import numpy as np

from alexdoor_xas.action.b1 import finite_vector
from alexdoor_xas.perception.contracts import TIME_ROUNDOFF_S
from alexdoor_xas.perception.load import (
    require_joint_output_feedback,
    residual_component_bounds,
)


class CausalResidualStopController:
    """Cancel only the certain residual; never change targets or infer missing load.

    The qualified force domain is 10 N. This response does not certify future
    stopping support, arbitrary disturbances or inputs outside the calibration.
    Preview is pure; observation accepts only completed robot-output samples.
    """

    def __init__(self, calibration):
        if calibration.verified is not True or calibration.stop_verified is not True:
            raise ValueError("unverified_robot_load_or_stop_recipe")
        if type(calibration.external_sign) is not int or calibration.external_sign not in (-1, 1):
            raise ValueError("unverified_feedback_sign")
        self.calibration = calibration
        tolerance, caps = calibration.position_support()
        paid = np.where(caps >= 10, tolerance, 0)
        lower = finite_vector(calibration.q_min, 7) - paid[0]
        upper = finite_vector(calibration.q_max, 7) + paid[1]
        # Preserve the already qualified float32 halo representation.
        self.lower = np.where(
            paid[0] > 0, np.nextafter(lower.astype(np.float32), np.float32(-np.inf)), lower
        )
        self.upper = np.where(
            paid[1] > 0, np.nextafter(upper.astype(np.float32), np.float32(np.inf)), upper
        )
        self.speed = finite_vector(calibration.speed_max_rad_s, 7)
        self.acceleration = finite_vector(calibration.acceleration_max_rad_s2, 7)
        bounds = finite_vector(calibration.residual_bound_nm, 7)
        if (
            (self.lower > self.upper).any()
            or (self.speed < 0).any()
            or (self.acceleration < 0).any()
        ):
            raise ValueError("invalid_stop_calibration_domain")
        if (residual_component_bounds(calibration, np.zeros(7), np.zeros(7), bounds) < 0).any():
            raise ValueError("invalid_stop_residual_bound")
        self.reset(-1)

    def reset(self, generation):
        """Cold start: a moving sample is never silently reused after the latch."""
        self.generation = generation
        self.last = None
        self.observations = 0

    def observe(self, sample, generation):
        if generation != self.generation:
            raise ValueError("stop_feedback_generation_changed")
        f = sample.feedback
        stamp, available = float(f.timestamp_s), float(f.available_s)
        if not np.isfinite([stamp, available]).all() or stamp > available:
            raise ValueError("invalid_stop_feedback_clock")
        record = dict(
            time_s=stamp,
            available_s=available,
            generation=generation,
            valid=False,
            origin=(f.source, f.semantics, f.units, f.sign_convention, f.healthy, f.reason),
            reason="unavailable_robot_feedback",
            residual=None,
            bound=None,
        )
        try:
            require_joint_output_feedback(f, self.calibration)
            q, dq, ddq, expected, feedback = [
                finite_vector(v, 7)
                for v in (sample.q, sample.dq, sample.ddq, sample.expected_torque_nm, f.torque_nm)
            ]
            inside = bool(
                (
                    (q >= self.lower)
                    & (q <= self.upper)
                    & (abs(dq) <= self.speed)
                    & (abs(ddq) <= self.acceleration)
                ).all()
            )
            bound = residual_component_bounds(
                self.calibration, dq, ddq, finite_vector(self.calibration.residual_bound_nm, 7)
            )
            residual = self.calibration.external_sign * (feedback - expected)
            if not np.isfinite(np.r_[residual, bound]).all():
                raise ValueError("invalid_stop_residual_or_bound")
            record.update(
                valid=inside,
                reason="" if inside else "outside_existing_load_calibration",
                residual=residual,
                bound=bound,
            )
        except ValueError as error:
            record["reason"] = str(error)
        if self.last is not None and stamp <= self.last["time_s"]:
            if stamp < self.last["time_s"]:
                raise ValueError("stop_feedback_time_reversed")
            if any(
                self.last[k] != record[k]
                for k in ("valid", "reason", "available_s", "generation", "origin")
            ) or any(not np.array_equal(self.last[k], record[k]) for k in ("residual", "bound")):
                raise ValueError("different_stop_feedback_at_same_time")
            return
        self.last = record
        self.observations += 1

    def preview(self, now, generation):
        if generation != self.generation:
            raise ValueError("stop_dispatch_generation_changed")
        if not np.isfinite(now):
            raise ValueError("invalid_stop_dispatch_clock")
        record, zero = self.last, np.zeros(7)
        if record is None:
            return zero, dict(valid=False, reason="no_completed_robot_sample", feedback_time_s=None)
        receipt = dict(valid=False, reason=record["reason"], feedback_time_s=record["time_s"])
        if not record["valid"]:
            return zero, receipt
        if not (
            record["available_s"] <= now + TIME_ROUNDOFF_S
            and abs(now - record["time_s"]) <= TIME_ROUNDOFF_S
        ):
            receipt["reason"] = "feedback_not_last_completed_interval"
            return zero, receipt
        residual, bound = record["residual"], record["bound"]
        correction = -np.sign(residual) * np.maximum(np.abs(residual) - bound, 0)
        return correction, dict(
            valid=True,
            reason="",
            feedback_time_s=record["time_s"],
            feedback_available_s=record["available_s"],
            residual_nm=residual.tolist(),
            component_bound_nm=bound.tolist(),
        )
