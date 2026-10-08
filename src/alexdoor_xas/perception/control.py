"""Conservative prototype control checks using observed geometry and robot proprioception."""

import numpy as np

from alexdoor_xas.perception.admission import require_current_admission
from alexdoor_xas.perception.contracts import LEGACY_FULL_STATE, geometry_profile, validate_geometry


class ObservedControlChecks:
    """RGB-D cannot establish loaded contact or force; never manufacture that feedback."""

    def __init__(self, setup):
        self.setup = setup
        self.reset()

    def reset(self):
        self.initial_tool = None

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
            else:
                validate_geometry(estimate, float(sensor["time_s"]), 0.15, profile=profile)
        except ValueError:
            return "invalid_perception"
        loaded = stage in ("contact", "push", "hold") or (
            admission is not None and admission.proposal.loaded
        )
        if loaded:
            normal = estimate.contact_rotation[:, 0]
            alignment = np.arccos(np.clip(tool.rot[:, 0] @ normal, -1, 1))
            if alignment > self.setup.orientation_tolerance:
                return "observed_contact_orientation"
            # A complete geometric estimate alone cannot certify the frozen force/load rules.
            if feedback is None or feedback.torque_nm is None:
                return "force_feedback_unavailable"
            return "load_control_unverified"
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
        return self.checks.check(
            {key: array(value) for key, value in io.observe().items()},
            getattr(self.provider, "last_estimate", None),
            io.tool_pose(),
            array(io.env.limits)[0],
            stage,
            array(io.env.targets)[0, io.env.neck_ids],
            profile=geometry_profile(self.provider.binding),
            feedback=io.robot_feedback(),
            admission=io.admission,
        )
