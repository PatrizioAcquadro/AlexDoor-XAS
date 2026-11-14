"""Motion checks using observed geometry, proprioception and bounded robot load."""

import numpy as np

from alexdoor_xas.perception.admission import require_current_admission
from alexdoor_xas.perception.contracts import LEGACY_FULL_STATE, validate_geometry


class ObservedControlChecks:
    """RGB-D cannot establish loaded contact or force; never manufacture that feedback."""

    def __init__(self, setup, *, contact_monitor=None, on_contact_evidence=None):
        self.setup = setup
        self.contact_monitor = contact_monitor
        self.on_contact_evidence = on_contact_evidence
        self.reset()

    def reset(self):
        self.initial_tool = None
        self.last_failure = None
        self.pending_unloading_orientation = None
        if self.contact_monitor is not None:
            self.contact_monitor.reset()

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
        defer_unloading_orientation=False,
    ):
        self.pending_unloading_orientation = None
        if admission is not None and profile == LEGACY_FULL_STATE:
            return "geometry_profile_mismatch"
        q = np.asarray(sensor["joint_position"])
        dq = np.asarray(sensor["joint_velocity"])
        if q.shape != (9,) or dq.shape != (9,) or not np.isfinite(np.r_[q, dq]).all():
            return "invalid_proprioception"
        if np.max(np.abs(dq[:7])) > self.setup.max_joint_speed * 1.02:
            return "actual_joint_speed_limit"
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
        except ValueError as error:
            self.last_failure = dict(
                time_s=float(sensor["time_s"]),
                reason=str(error),
                decision=getattr(error, "decision", None),
            )
            return "invalid_perception"
        loaded = stage in ("contact", "push", "hold") or (
            admission is not None and admission.proposal.loaded
        )
        if loaded:
            normal_alignment = None
            if estimate.local is not None and (
                estimate.operational is None
                or admission is not None
                and admission.local_evidence is not None
            ):
                from alexdoor_xas.perception.contracts import validate_local_contact

                patch = validate_local_contact(estimate.local, float(sensor["time_s"]))
                contact = patch.world_pose
                support = patch.pose_support
                normal_bound = patch.normal_rotation_bound_rad
                normal_alignment = patch.normal_alignment
            else:
                from alexdoor_xas.action.frames import ObjectFrame

                normal_bound = None
                contact = ObjectFrame(estimate.contact_position, estimate.contact_rotation)
                support = (
                    estimate.operational.contact.world_support
                    if estimate.operational is not None
                    else None
                )
            normal = contact.rot[:, 0]
            alignment = np.arccos(np.clip(tool.rot[:, 0] @ normal, -1, 1))
            orientation_bound = (
                normal_bound
                if normal_bound is not None
                else support.rotation_bound_rad
                if support is not None
                else 0
            )
            if normal_alignment is not None:
                # Each cap bounds the same feasible normal. Keep the tightest
                # bound on tool alignment, including the original local bound.
                alignment, orientation_bound = normal_alignment.tool_bound(
                    tool.rot[:, 0], (alignment, orientation_bound)
                )
            robot_rotation_bound = 0.0
            bound_source = "no_admission"
            if admission is not None:
                robot_rotation_bound = admission.limits.robot_rotation_bound_rad
                bound_source = "controller_response_fallback"
                if self.contact_monitor is not None:
                    measured = self.contact_monitor.config.robot_fk_rotation_bound_rad
                    if measured is not None:
                        # `tool` is current proprioceptive FK. Future controller
                        # response remains charged by motion admission; it is
                        # not an error in this already measured current pose.
                        robot_rotation_bound = measured
                        bound_source = "calibrated_current_fk"
                orientation_bound += robot_rotation_bound
            orientation_rejected = alignment + orientation_bound > self.setup.orientation_tolerance
            orientation_failure = dict(
                time_s=float(sensor["time_s"]),
                reason="observed_contact_orientation",
                alignment_rad=float(alignment),
                geometry_rotation_bound_rad=float(orientation_bound - robot_rotation_bound),
                current_robot_rotation_bound_rad=float(robot_rotation_bound),
                robot_bound_source=bound_source,
                tolerance_rad=float(self.setup.orientation_tolerance),
            )
            if orientation_rejected and stage in ("contact", "push", "hold"):
                self.last_failure = orientation_failure
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
            )
            if not result.admitted:
                return result.reason
            now = float(sensor["time_s"])
            if (
                not loaded
                and self.contact_monitor is not None
                and load.normal_lower_n >= self.contact_monitor.config.force_on_n
            ):
                return "unexpected_observed_load"
            # The runner checks health before creating its first proposal. No
            # contact is admitted there; receipt-dependent checks run before
            # actual dispatch and after each step.
            if loaded or self.contact_monitor is not None and admission is not None:
                if self.contact_monitor is None:
                    return "operational_contact_monitor_unconfigured"
                evidence = self.contact_monitor.update(now, estimate, tool, load, admission)
                if self.on_contact_evidence is not None:
                    self.on_contact_evidence(evidence)
                if evidence.status in ("lost", "unavailable"):
                    return evidence.reason
                if loaded and orientation_rejected and evidence.status != "separated":
                    if (
                        defer_unloading_orientation
                        and self.contact_monitor.elastic_unloading
                        and evidence.status in ("contact", "unloading")
                    ):
                        # Only the dispatcher can clear this after checking its
                        # exact pose request, controller target and sweep.
                        self.pending_unloading_orientation = orientation_failure
                    else:
                        self.last_failure = orientation_failure
                        return "observed_contact_orientation"
                if stage in ("push", "hold") and evidence.status != "contact":
                    return "observed_contact_not_established"
        return ""


class ObservedPurdueSafety:
    """Purdue IO adapter; reads neither the door articulation nor prepared geometry."""

    def __init__(self, io, setup, provider, *, contact_monitor=None):
        self.io, self.provider = io, provider
        self.gripper_ids = None
        self.checks = ObservedControlChecks(setup, contact_monitor=contact_monitor)

    def reset(self):
        self.checks.reset()

    def before_command(self, stage):
        return self.check(stage, defer_unloading_orientation=True)

    def check(self, stage, *, defer_unloading_orientation=False):
        from alexdoor_xas.recording.b1_runtime import array

        io = self.io
        if any(
            np.max(np.abs(sample.dq)) > self.checks.setup.max_joint_speed * 1.02
            for sample in io.env.load_samples
        ):
            return "actual_joint_speed_limit"
        if self.gripper_ids is None:
            names = [
                f"{side}_WSG32_{joint}"
                for side in ("left", "right")
                for joint in ("JAW_OPENING", "JAW_FOLLOWER")
            ]
            try:
                ids, found = io.env.robot.find_joints(names, preserve_order=True)
            except (AttributeError, ValueError):
                return "missing_gripper_proprioception"
            if tuple(found) != tuple(names):
                return "missing_gripper_proprioception"
            self.gripper_ids = ids
        grippers = array(io.env.robot.data.joint_pos)[0, self.gripper_ids]
        if not np.isfinite(grippers).all() or np.max(np.abs(grippers)) > 0.001:
            return "gripper_opened"
        feedback = io.robot_feedback()
        try:
            load = (
                io.load_source(io, feedback)
                if io.load_source is not None and stage != "inspect"
                else None
            )
        except (ValueError, RuntimeError, TypeError, AttributeError) as error:
            self.checks.last_failure = dict(
                time_s=feedback.timestamp_s,
                reason=str(error),
                decision=getattr(error, "decision", None),
            )
            return f"load_estimation_failed: {error}"
        return self.checks.check(
            {key: array(value) for key, value in io.observe().items()},
            getattr(self.provider, "last_estimate", None),
            io.tool_pose(),
            array(io.env.limits)[0],
            stage,
            array(io.env.targets)[0, io.env.neck_ids],
            profile=io.profile,
            feedback=feedback,
            load=load,
            admission=io.admission,
            defer_unloading_orientation=defer_unloading_orientation,
        )
