"""Independent learned-pilot scoring; no result is supplied to deployed control."""

import numpy as np
from scipy.spatial.transform import Rotation

from alexdoor_xas.qualification.synthetic_probe import ContactLoad, SustainedAngle


class PilotEvaluation:
    """Frozen physical criteria, controlled sustain, complete hold and commanded release.

    The caller records these rows after execution. This object has no controller,
    stop callback, policy input or access to the operational admission producer.
    """

    def __init__(self, setup, expert_degrees, control_dt):
        if not np.isfinite(expert_degrees) or expert_degrees <= 0:
            raise ValueError("Missing frozen matched expert reference")
        self.setup, self.dt = setup, control_dt
        self.reference_deg = float(expert_degrees)
        self.required_angle = np.deg2rad(0.8 * expert_degrees)
        self.load = ContactLoad(
            setup.contact_force_window_s,
            setup.loaded_force_n,
            setup.contact_gap_tolerance_m,
        )
        self.window = SustainedAngle(setup.sustain_s)
        self.hold_start = self.last_time = None
        self.hold_valid = self.released = self.release_attempted = False
        self.was_loaded = False
        self.contact_losses = self.learned_commands = self.reference_commands = 0
        self.violations = set()
        self.peak_force_n = 0.0
        self.raw_max_angle_rad = 0.0
        self.last_row = None

    def update(self, row):
        t = float(row["time_s"])
        if self.last_time is not None and not np.isclose(t - self.last_time, self.dt, atol=1e-8):
            raise ValueError("Independent evaluation clock gap")
        self.last_time = t
        force = row["native_force_n"]
        gaps = row["native_gap_m"]
        if len(force) != 2 or len(gaps) != 2 or not np.isfinite(force).all():
            raise ValueError("Missing two native physical samples")
        gap = max(gaps) if all(v is not None for v in gaps) else None
        loaded = self.load.update(t, float(np.mean(force)), gap)
        self.peak_force_n = max(self.peak_force_n, max(force, default=0.0))
        self.violations.update(row["physical_violations"])
        self.learned_commands += int(bool(row["learned_command"]))
        self.reference_commands += int(bool(row.get("reference_command", False)))
        commanded = bool(row["learned_command"] or row.get("reference_command", False))
        valid = bool(
            loaded
            and not row["physical_violations"]
            and row["footprint_inside"]
            and row["material_error_m"] <= self.setup.position_tolerance
            and row["material_orientation_error_rad"] <= self.setup.orientation_tolerance
            and commanded
        )
        angle = float(row["opening_angle_rad"])
        self.raw_max_angle_rad = max(self.raw_max_angle_rad, angle)
        # Active, physically loaded material tracking remains controlled contact;
        # uncommanded or unloaded coasting never contributes to this window.
        self.window.update(t, angle, valid)
        if valid and angle >= self.required_angle:
            self.hold_start = t - self.dt if self.hold_start is None else self.hold_start
            if t - self.hold_start >= self.setup.hold_settle_s - 1e-9:
                self.hold_valid = True
        else:
            self.hold_start = None
        withdrawing = commanded and row["commanded_withdrawal"]
        self.release_attempted |= self.hold_valid and self.was_loaded and withdrawing
        released_now = bool(
            self.hold_valid
            and self.release_attempted
            and commanded
            and not loaded
            and row["surface_separation_m"] is not None
            and row["surface_separation_m"] >= 0.01
            and row["applied_tracking_error_m"] is not None
            and row["applied_tracking_error_m"] <= self.setup.position_tolerance
            and not row["physical_violations"]
        )
        if self.released and loaded:
            self.violations.add("recontact_after_release")
        self.released |= released_now
        if (
            self.was_loaded
            and not loaded
            and not self.release_attempted
            and self.window.maximum is not None
        ):
            self.contact_losses += 1
        self.was_loaded = loaded
        self.last_row = dict(
            row, loaded=loaded, valid_contact=valid, filtered_force_n=self.load.force
        )
        return self.last_row

    def result(self):
        angle = 0.0 if self.window.maximum is None else float(self.window.maximum)
        complete = bool(
            angle >= self.required_angle
            and self.hold_valid
            and self.released
            and not self.violations
            and self.contact_losses == 0
        )
        return dict(
            task_success=bool(self.learned_commands and not self.reference_commands and complete),
            reference_cycle_success=bool(
                self.reference_commands and not self.learned_commands and complete
            ),
            reference_commands=self.reference_commands,
            raw_max_opening_deg=float(np.rad2deg(self.raw_max_angle_rad)),
            controlled_opening_deg=float(np.rad2deg(angle)),
            expert_reference_deg=self.reference_deg,
            controlled_fraction_of_expert=float(np.rad2deg(angle) / self.reference_deg),
            hold_valid=self.hold_valid,
            released=self.released,
            release_attempted=self.release_attempted,
            contact_losses=self.contact_losses,
            learned_commands=self.learned_commands,
            peak_native_force_n=self.peak_force_n,
            physical_violations=sorted(self.violations),
        )


def physical_row(
    env, door, setup, time_s, applied_goal, *, learned_command, reference_command=False
):
    """Evaluator-only extraction. Prepared geometry never returns through robot IO."""
    from alexdoor_xas.recording.b1_runtime import array

    position, quaternion = [array(v)[0] for v in env.tool_pose()]
    rotation = Rotation.from_quat(quaternion).as_matrix()
    angle = float(array(env.door.data.joint_pos)[0, 0])
    material_p, material_r = door.contact_pose(angle, setup.contact_fraction, setup.contact_height)
    force, gaps, violations = [], [], []
    for sample in env.contact_history:
        intended = [c for c in sample["contacts"] if c["category"] in ("positive", "negative")]
        force.append(float(sum(c["force_n"] for c in intended)))
        gaps.append(min((c["separation_m"] for c in intended), default=None))
        if sample["forbidden"]:
            violations.append("forbidden_contact")
    q = array(env.robot.data.joint_pos)[0]
    limits = array(env.limits)[0]
    arm = q[env.arm_ids]
    if np.any(arm < limits[:, 0] - 0.005) or np.any(arm > limits[:, 1] + 0.005):
        violations.append("joint_limit_violation")
    if any(np.max(abs(s.dq)) > setup.max_joint_speed * 1.02 for s in env.load_samples):
        violations.append("actual_joint_speed_limit")
    if max(force, default=0) > setup.force_limit_n:
        violations.append("force_limit")
    names = [
        f"{side}_WSG32_{joint}"
        for side in ("left", "right")
        for joint in ("JAW_OPENING", "JAW_FOLLOWER")
    ]
    ids, _ = env.robot.find_joints(names, preserve_order=True)
    if np.max(abs(q[ids])) > 0.001:
        violations.append("gripper_opened")
    if not np.isfinite(np.r_[q, position, rotation.ravel(), angle, force]).all():
        violations.append("invalid_physics")
    panel_rotation = door.rotation(angle)
    closed = panel_rotation.T @ (position - door.hinge) + door.hinge
    try:
        front_x, outward = door.front_surface(float(closed[1]), float(closed[2]))
        normal = panel_rotation @ (-outward / np.linalg.norm(outward))
        front = (
            panel_rotation @ (np.array([front_x, closed[1], closed[2]]) - door.hinge) + door.hinge
        )
        separation = float((front - position) @ normal)
        withdrawal = 0.0 if applied_goal is None else float((front - applied_goal.origin) @ normal)
    except ValueError:
        separation, withdrawal = None, None
    return dict(
        time_s=float(time_s),
        opening_angle_rad=abs(angle),
        native_force_n=force,
        native_gap_m=gaps,
        physical_violations=sorted(set(violations)),
        footprint_inside=bool(
            door.footprint_inside(
                np.concatenate(env.push_geometry.distal_faces) @ rotation.T + position,
                angle,
            )
        ),
        material_error_m=float(np.linalg.norm(position - material_p)),
        material_orientation_error_rad=float(
            Rotation.from_matrix(material_r @ rotation.T).magnitude()
        ),
        surface_separation_m=separation,
        applied_withdrawal_m=withdrawal,
        commanded_withdrawal=bool(
            applied_goal is not None
            and separation is not None
            and (position - applied_goal.origin) @ normal
            > 32 * np.finfo(np.float32).eps * max(1.0, np.linalg.norm(position))
        ),
        applied_tracking_error_m=(
            None if applied_goal is None else float(np.linalg.norm(applied_goal.origin - position))
        ),
        learned_command=bool(learned_command),
        reference_command=bool(reference_command),
    )
