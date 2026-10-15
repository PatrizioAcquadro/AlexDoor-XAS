"""Conservative prototype control checks using observed geometry and robot proprioception."""

from collections import deque

import numpy as np

from alexdoor_xas.perception.admission import require_current_admission, rotational_travel
from alexdoor_xas.perception.contracts import LEGACY_FULL_STATE, geometry_profile, validate_geometry


class ObservedControlChecks:
    """RGB-D cannot establish loaded contact or force; never manufacture that feedback."""

    def __init__(self, setup):
        self.setup = setup
        self.reset()

    def reset(self):
        self.initial_tool = None
        self.force_history = deque()

    def check(
        self,
        sensor,
        estimate,
        tool,
        limits,
        stage,
        neck_target,
        *,
        profile=LEGACY_FULL_STATE,
        feedback=None,
        admission=None,
        load=None,
    ):
        if admission is not None and profile == LEGACY_FULL_STATE:
            return "geometry_profile_mismatch"
        q = np.asarray(sensor["joint_position"])
        dq = np.asarray(sensor["joint_velocity"])
        if q.shape != (9,) or dq.shape != (9,) or not np.isfinite(np.r_[q, dq]).all():
            return "invalid_proprioception"
        if np.any(q[:7] < limits[:, 0] - 0.005) or np.any(q[:7] > limits[:, 1] + 0.005):
            return "joint_limit_violation"
        if not np.isfinite(tool.origin).all() or not np.isfinite(tool.rot).all():
            return "invalid_robot_fk"
        if not np.any(sensor["valid_depth"]) or not np.any(sensor["rgb"]):
            return "missing_rgbd"
        if self.initial_tool is None:
            self.initial_tool = tool.origin.copy()
        if stage == "inspect":
            if (
                np.linalg.norm(tool.origin - self.initial_tool) > 0.01
                or np.abs(q[7:] - neck_target).max() > 0.1
            ):
                return "unsafe_inspection"
            return ""
        if estimate is None:
            return "invalid_perception"
        try:
            if admission is not None:
                require_current_admission(admission, estimate, float(sensor["time_s"]))
            elif estimate.local is not None:
                from alexdoor_xas.perception.contracts import validate_local_contact

                validate_local_contact(estimate.local, float(sensor["time_s"]))
            else:
                validate_geometry(estimate, float(sensor["time_s"]), 0.15, profile=profile)
        except ValueError:
            return "invalid_perception"
        loaded = stage in ("contact", "push", "hold") or (
            admission is not None and admission.proposal.loaded
        )
        if loaded:
            if estimate.local is not None and (
                estimate.operational is None
                or admission is not None
                and admission.local_evidence is not None
            ):
                from alexdoor_xas.perception.contracts import validate_local_contact

                patch = validate_local_contact(estimate.local, float(sensor["time_s"]))
                contact = patch.world_pose
                support = patch.pose_support
            else:
                from alexdoor_xas.action.frames import ObjectFrame

                contact = ObjectFrame(estimate.contact_position, estimate.contact_rotation)
                support = (
                    estimate.operational.contact.world_support
                    if estimate.operational is not None
                    else None
                )
            normal = contact.rot[:, 0]
            alignment = np.arccos(np.clip(tool.rot[:, 0] @ normal, -1, 1))
            orientation_bound = support.rotation_bound_rad if support is not None else 0
            if admission is not None:
                orientation_bound += admission.limits.robot_rotation_bound_rad
            if alignment + orientation_bound > self.setup.orientation_tolerance:
                return "observed_contact_orientation"
            # A complete geometric estimate alone cannot certify the frozen force/load rules.
            if feedback is None or feedback.torque_nm is None:
                return "force_feedback_unavailable"
            if load is None:
                return "load_control_unverified"
        if load is not None:
            if feedback is None or (
                load.feedback.timestamp_s != feedback.timestamp_s
                or load.feedback.available_s != feedback.available_s
                or load.feedback.source != feedback.source
                or load.feedback.semantics != feedback.semantics
                or not np.array_equal(load.feedback.torque_nm, feedback.torque_nm)
            ):
                return "load_feedback_mismatch"
            result = load.admission(
                float(sensor["time_s"]),
                self.setup.force_limit_n,
                require_contact=False,
                loaded_force_n=self.setup.loaded_force_n,
            )
            if not result.admitted:
                return result.reason
            now = float(sensor["time_s"])
            if not self.force_history or self.force_history[-1][0] != now:
                self.force_history.append((now, load.normal_lower_n))
            while now - self.force_history[0][0] >= self.setup.contact_force_window_s - 1e-9:
                self.force_history.popleft()
            mean_lower = sum(value for _, value in self.force_history) / len(self.force_history)
            if not loaded and mean_lower >= self.setup.loaded_force_n:
                return "unexpected_observed_load"
            if stage in ("push", "hold"):
                if support is None or admission is None:
                    return "unverified_contact_distance"
                gap_upper = (
                    abs(normal @ (tool.origin - contact.origin))
                    + support.position_bound_m
                    + rotational_travel(
                        np.linalg.norm(tool.origin - contact.origin), support.rotation_bound_rad
                    )
                    + admission.limits.robot_position_bound_m
                )
                if (
                    mean_lower < self.setup.loaded_force_n
                    or gap_upper > self.setup.contact_gap_tolerance_m
                ):
                    return "uncontrolled_observed_contact"
        return ""


class ObservedPurdueSafety:
    """Purdue IO adapter; reads neither the door articulation nor prepared geometry."""

    def __init__(self, io, setup, provider):
        self.io, self.provider = io, provider
        self.checks = ObservedControlChecks(setup)

    def reset(self):
        self.checks.reset()

    def before_command(self, stage):
        return self.check(stage)

    def check(self, stage):
        from alexdoor_xas.recording.b1_runtime import array

        io = self.io
        feedback = io.robot_feedback()
        try:
            load = (
                io.load_source(io, feedback)
                if io.load_source is not None and stage != "inspect"
                else None
            )
        except (ValueError, RuntimeError, TypeError, AttributeError) as error:
            return f"load_estimation_failed: {error}"
        return self.checks.check(
            {key: array(value) for key, value in io.observe().items()},
            getattr(self.provider, "last_estimate", None),
            io.tool_pose(),
            array(io.env.limits)[0],
            stage,
            array(io.env.targets)[0, io.env.neck_ids],
            profile=geometry_profile(self.provider.binding),
            feedback=feedback,
            load=load,
            admission=io.admission,
        )
