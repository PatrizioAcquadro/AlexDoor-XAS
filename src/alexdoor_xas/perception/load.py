"""Robot-only residual wrench bounds; calibration/stop qualification is explicit."""

from dataclasses import dataclass, replace

import numpy as np

from alexdoor_xas.action.b1 import finite_vector
from alexdoor_xas.assets.purdue import ARM_JOINTS
from alexdoor_xas.perception.admission import admit_load
from alexdoor_xas.perception.contracts import RobotFeedback


def simulated_feedback(env, *, time_s=None):
    """Read solver joint-force projection, never effort targets or applied_torque.

    PhysX projects incoming joint spatial forces along each DOF's motion axis.
    This is joint-output feedback, not external/contact force. Its sign and
    residual model must be verified independently before loaded control.
    """
    from alexdoor_xas.recording.b1_runtime import array

    now = float(env._time_s if hasattr(env, "_time_s") else env.capture.sample.time_s)
    if time_s is not None:
        now = float(time_s)
    try:
        values = array(env.robot.root_view.get_dof_projected_joint_forces())[0, env.arm_ids]
        values = finite_vector(values, 7).copy()
    except (AttributeError, RuntimeError, ValueError):
        return RobotFeedback.unavailable(now, ARM_JOINTS)
    return RobotFeedback(
        now, now, ARM_JOINTS, values, "simulated_actuator_feedback", "joint_output", True
    )


@dataclass(frozen=True)
class LoadCalibration:
    """A bounded recipe supplied by robot-only dynamic and stop verification.

    Bounds must include dynamics, friction, acceleration estimation and feedback
    error over the supplied joint envelope. No fit on task contact truth at run time.
    """

    residual_bound_nm: np.ndarray
    q_min: np.ndarray
    q_max: np.ndarray
    speed_max_rad_s: np.ndarray
    acceleration_max_rad_s2: np.ndarray
    external_sign: int
    verified: bool = False
    stop_verified: bool = False
    velocity_residual_nm_per_rad_s: np.ndarray | None = None
    acceleration_residual_nm_per_rad_s2: np.ndarray | None = None
    tool_normal_residual_n: np.ndarray | None = None  # constant, max |dq|, max |ddq|
    tool_normal_force_upper_max_n: float | None = None


@dataclass(frozen=True)
class LoadEstimate:
    feedback: RobotFeedback
    wrench: np.ndarray
    force_upper_n: float
    normal_lower_n: float
    residual_error_nm: np.ndarray
    residual_bound_nm: np.ndarray
    model_verified: bool
    stop_verified: bool
    normal_samples: tuple = ()  # (interval_start_s, interval_end_s, mean_lower_n)

    def admission(self, now, force_limit_n, *, require_contact=False, loaded_force_n=0):
        return admit_load(
            self.feedback,
            now,
            ARM_JOINTS,
            timeout_s=2 / 60,
            model_semantics="joint_output",
            residual_bound_nm=self.residual_bound_nm,
            load_upper_n=self.force_upper_n,
            force_limit_n=force_limit_n,
            model_verified=self.model_verified,
            stop_verified=self.stop_verified,
            contact_consistent=(not require_contact or self.normal_lower_n >= loaded_force_n),
        )


class ResidualLoadModel:
    def __init__(self, calibration):
        self.calibration = calibration

    def estimate(
        self,
        feedback,
        expected_torque_nm,
        jacobian_world,
        q,
        dq,
        ddq,
        normal_world,
        *,
        tool_normal_world=None,
    ):
        """J.T maps an external world wrench at the distal tool into joint torque.

        Full column rank is required. The pseudoinverse interval bound includes
        every calibrated joint residual; inconsistent residuals and envelope exits
        fail closed. Force norm includes tangential load; torque is not called force.
        """
        c = self.calibration
        if (
            feedback.joint_names != ARM_JOINTS
            or feedback.semantics != "joint_output"
            or feedback.source not in ("reported_joint_state", "simulated_actuator_feedback")
            or feedback.units != "N m"
            or feedback.sign_convention != "positive_joint_coordinate"
            or feedback.healthy is not True
            or feedback.reason
        ):
            raise ValueError("unverified_torque_semantics")
        if type(c.external_sign) is not int or c.external_sign not in (-1, 1):
            raise ValueError("unverified_feedback_sign")
        q, dq, ddq, bounds = [finite_vector(v, 7) for v in (q, dq, ddq, c.residual_bound_nm)]
        if (
            np.any(bounds < 0)
            or np.any(q < finite_vector(c.q_min, 7))
            or np.any(q > finite_vector(c.q_max, 7))
            or np.any(np.abs(dq) > finite_vector(c.speed_max_rad_s, 7))
            or np.any(np.abs(ddq) > finite_vector(c.acceleration_max_rad_s2, 7))
        ):
            raise ValueError("outside_verified_load_envelope")
        # Dynamics couple the arm: a stationary joint can carry residual error
        # from another joint's motion. Use whole-arm norms, not its own speed.
        for coefficient, magnitude in (
            (c.velocity_residual_nm_per_rad_s, np.max(np.abs(dq))),
            (c.acceleration_residual_nm_per_rad_s2, np.max(np.abs(ddq))),
        ):
            if coefficient is not None:
                coefficient = finite_vector(coefficient, 7)
                if np.any(coefficient < 0):
                    raise ValueError("invalid_dynamic_residual_bound")
                bounds = bounds + coefficient * magnitude
        jac = np.asarray(jacobian_world, dtype=float)
        if jac.shape != (6, 7) or not np.isfinite(jac).all():
            raise ValueError("invalid_load_jacobian")
        matrix = jac.T
        if np.linalg.matrix_rank(matrix) != 6:
            raise ValueError("unobservable_external_wrench")
        inverse = np.linalg.pinv(matrix)
        residual = c.external_sign * (
            finite_vector(feedback.torque_nm, 7) - finite_vector(expected_torque_nm, 7)
        )
        wrench = inverse @ residual
        error = residual - matrix @ wrench
        # Feasible residual uncertainty can amplify under the projection.
        if np.any(np.abs(error) > np.abs(np.eye(7) - matrix @ inverse) @ bounds + 1e-9):
            raise ValueError("inconsistent_single_tool_wrench")
        radius = np.abs(inverse) @ bounds
        normal = finite_vector(normal_world, 3)
        if not np.isclose(np.linalg.norm(normal), 1, atol=1e-6):
            raise ValueError("invalid_load_normal")
        normal_lower = -normal @ wrench[:3] - np.abs(normal) @ radius[:3]
        upper = np.linalg.norm(wrench[:3]) + np.linalg.norm(radius[:3])
        if c.tool_normal_residual_n is None and c.tool_normal_force_upper_max_n is not None:
            raise ValueError("invalid_tool_normal_calibration")
        if c.tool_normal_residual_n is not None:
            coefficients = finite_vector(c.tool_normal_residual_n, 3)
            maximum = c.tool_normal_force_upper_max_n
            if (
                np.any(coefficients < 0)
                or maximum is None
                or not np.isfinite(maximum)
                or maximum <= 0
            ):
                raise ValueError("invalid_tool_normal_calibration")
            # The scalar empirical envelope has an explicitly bounded weak-load
            # domain. Stronger loads retain the original joint-box wrench bound.
            if upper <= maximum:
                tool_normal = finite_vector(tool_normal_world, 3)
                if not np.isclose(np.linalg.norm(tool_normal), 1, atol=1e-6):
                    raise ValueError("invalid_robot_tool_normal")
                normal_error = coefficients @ [1.0, np.max(np.abs(dq)), np.max(np.abs(ddq))]
                normal_error += np.linalg.norm(normal - tool_normal) * np.linalg.norm(radius[:3])
                normal_lower = max(normal_lower, -normal @ wrench[:3] - normal_error)

        return LoadEstimate(
            feedback,
            wrench,
            float(upper),
            float(normal_lower),
            error,
            bounds,
            c.verified,
            c.stop_verified,
        )


class PurdueLoadSource:
    """Runtime adapter for a supplied verified robot calibration; no task truth.

    Solver mass/gravity/Coriolis are robot model quantities. Unmodelled friction,
    finite-difference acceleration and solver errors belong in calibration bounds.
    Without that evidence the resulting load admission remains unverified.
    """

    def __init__(self, calibration, *, normal_angular_speed_rad_s):
        if not np.isfinite(normal_angular_speed_rad_s) or normal_angular_speed_rad_s < 0:
            raise ValueError("unmeasured_material_normal_rate")
        self.normal_angular_speed_rad_s = float(normal_angular_speed_rad_s)
        self.model = ResidualLoadModel(calibration)

    def __call__(self, io, feedback):
        from alexdoor_xas.perception.contracts import validate_local_contact
        from alexdoor_xas.recording.b1_runtime import array

        env = io.env
        estimate = io.safety.provider.last_estimate
        local = estimate.local is not None and (
            estimate.operational is None
            or io.admission is not None
            and io.admission.local_evidence is not None
        )
        if local:
            patch = validate_local_contact(estimate.local, feedback.timestamp_s)
            normal, support = patch.world_pose.rot[:, 0], patch.pose_support
            normal_rotation = patch.normal_rotation_bound_rad
        else:
            normal = estimate.contact_rotation[:, 0]
            support = estimate.operational.contact.world_support
            normal_rotation = None
        support.require_bounds(rotation=True)
        # Project force onto the independently supported normal. Tangential
        # frame ambiguity still belongs to finite material/sweep admission.
        normal_rotation = support.rotation_bound_rad if normal_rotation is None else normal_rotation
        if not np.isfinite(normal_rotation) or normal_rotation < 0:
            raise ValueError("invalid_load_normal_bound")
        samples = env.load_samples
        times = [sample.feedback.timestamp_s for sample in samples]
        if (
            len(samples) != 2
            or times[-1] != feedback.timestamp_s
            or not np.isclose(times[-1] - times[0], 1 / 120, atol=1e-9, rtol=0)
        ):
            raise ValueError("missing_physics_load_samples")
        neck = (
            array(env.capture.sample.joint_position)[0, 7:].copy()
            if self.model.calibration.tool_normal_residual_n is not None
            else None
        )
        loads = [
            self.model.estimate(
                sample.feedback,
                sample.expected_torque_nm,
                sample.jacobian_world,
                sample.q,
                sample.dq,
                sample.ddq,
                normal,
                tool_normal_world=(
                    io.tool_fk(
                        np.r_[sample.q, neck],
                        io.calibration,
                    ).rot[:, 0]
                    if self.model.calibration.tool_normal_residual_n is not None
                    else None
                ),
            )
            for sample in samples
        ]
        if any(not load.feedback.healthy or load.feedback.torque_nm is None for load in loads):
            raise ValueError("unhealthy_physics_feedback")
        # Retain the current measured feedback identity, and monitor the peak
        # bound across both 120 Hz substeps rather than only the 60 Hz endpoint.
        return replace(
            loads[-1],
            feedback=feedback,
            force_upper_n=max(load.force_upper_n for load in loads),
            normal_lower_n=min(
                load.normal_lower_n
                - load.force_upper_n
                * 2
                * np.sin(
                    min(
                        normal_rotation
                        + self.normal_angular_speed_rad_s
                        * (feedback.timestamp_s - load.feedback.timestamp_s),
                        np.pi,
                    )
                    / 2
                )
                for load in loads
            ),
            normal_samples=tuple(
                (
                    load.feedback.timestamp_s - 1 / 120,
                    load.feedback.timestamp_s,
                    load.normal_lower_n
                    - load.force_upper_n
                    * 2
                    * np.sin(
                        min(
                            normal_rotation
                            + self.normal_angular_speed_rad_s
                            * (feedback.timestamp_s - load.feedback.timestamp_s),
                            np.pi,
                        )
                        / 2
                    ),
                )
                for load in loads
            ),
        )


@dataclass(frozen=True)
class RobotLoadSample:
    feedback: RobotFeedback
    expected_torque_nm: np.ndarray
    jacobian_world: np.ndarray
    q: np.ndarray
    dq: np.ndarray
    ddq: np.ndarray


def simulated_load_sample(env, time_s):
    """Capture robot-only feedback/dynamics after a physics substep."""
    from alexdoor_xas.recording.b1_runtime import array

    acceleration = array(env.robot.data.joint_acc)[0]
    expected = (
        array(env.robot.data.mass_matrix)[0] @ acceleration
        + array(env.robot.data.gravity_compensation_forces)[0]
        + array(env.robot.root_view.get_coriolis_and_centrifugal_compensation_forces())[0]
    )
    return RobotLoadSample(
        simulated_feedback(env, time_s=time_s),
        expected[env.arm_ids].copy(),
        array(env.tool_jacobian())[0].copy(),
        array(env.robot.data.joint_pos)[0, env.arm_ids].copy(),
        array(env.robot.data.joint_vel)[0, env.arm_ids].copy(),
        acceleration[env.arm_ids].copy(),
    )
