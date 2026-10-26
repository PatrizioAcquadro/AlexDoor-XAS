"""Bounded executed disturbances and measured-response bookkeeping for short tasks."""

import numpy as np
from scipy.spatial.transform import Rotation

# These diagnostic annotations never enter observation streams.
EXPERT, INJECTED, CORRECTIVE = 0, 1, 2
NORMAL_RETREAT_M = {"A2": 0.015, "B1": 0.003, "B2": 0.0015, "C1": 0.001}


class CollectionCorrection:
    def __init__(
        self,
        family,
        variation,
        setup,
        dt,
        *,
        corrective_transition_s=None,
        approach_disturbance=None,
        corrective_brake_s=0.0,
    ):
        if family not in ("A", "B", "C", "D"):
            raise ValueError("Unknown phase family")
        self.family, self.variation, self.setup, self.dt = family, variation, setup, dt
        self.counts = {}
        self.role = EXPERT
        self.pending = False
        self.injected_commands = self.corrective_commands = 0
        self.recovered_samples = 0
        self.contact_loss_detected = False
        self.corrective_origin = None
        self.corrective_elapsed_s = 0.0
        self.corrective_transition_complete = False
        # C1's .8 s weakening pulse leaves only .25 s before the unchanged
        # recontact deadline. Its measured .93 mm normal return peaks at
        # 5.86 mm/s over .25 s, within the ordinary 6 mm/s gap catch-up rate.
        self.corrective_transition_s = (
            0.25 if family == "C" and variation == "correction-1" else 0.5
        )
        if corrective_transition_s is not None:
            value = float(corrective_transition_s)
            if (
                not np.isfinite(value)
                or not 0 < value <= (setup.approach_s if family == "A" else 1.0)
                or not (family == "A" or family == "C" and variation == "correction-2")
            ):
                raise ValueError("Return must fit its approach or one-second recontact envelope")
            self.corrective_transition_s = value
        self.corrective_brake_s = float(corrective_brake_s)
        if (
            not np.isfinite(self.corrective_brake_s)
            or self.corrective_brake_s < 0
            or (self.corrective_brake_s and family != "A")
        ):
            raise ValueError("A finite nonnegative brake is supported for approach corrections")
        self.corrective_start_error = None
        self.corrective_latest_error = None
        self.pre_action_reference = None
        self.reference_advance_rad = 0.0
        self.approach_disturbance = approach_disturbance
        self.pause_pose = None
        if approach_disturbance is not None:
            d = approach_disturbance
            if family != "A" or d["kind"] not in ("pause", "pose"):
                raise ValueError("Approach disturbances require an A pause or pose case")
            values = np.asarray(
                [
                    d["start_s"],
                    d["end_s"],
                    *d.get("translation_m", [0, 0, 0]),
                    *d.get("rotation_rad", [0, 0, 0]),
                ],
                dtype=float,
            )
            if (
                not np.isfinite(values).all()
                or not 0 <= d["start_s"] < d["end_s"]
                or d["end_s"] + self.corrective_brake_s + self.corrective_transition_s
                >= setup.approach_s
            ):
                raise ValueError("Approach disturbance must reserve its complete measured return")
            if (
                len(d.get("translation_m", [0, 0, 0])) != 3
                or len(d.get("rotation_rad", [0, 0, 0])) != 3
            ):
                raise ValueError("Approach disturbance needs three-axis pose offsets")
        self.had_contact = False
        self.unloaded_s = 0.0
        self.supported_hold_handoffs = 0

    def elapsed(self, phase):
        return self.counts.get(phase, 0) * self.dt

    def pulse(self, phase):
        if not self.variation.startswith("correction-"):
            return 0.0
        if self.approach_disturbance is not None:
            d = self.approach_disturbance
            t = self.elapsed(phase)
            if phase != "approach" or not d["start_s"] <= t < d["end_s"]:
                return 0.0
            if d["kind"] == "pause":
                return 1.0
            return float(np.clip((t - d["start_s"]) / 0.25, 0, 1))
        key = self.family + self.variation[-1]
        stage, start, end = {
            "A1": ("approach", 4.0, 6.0),
            "A2": ("approach", 4.0, 6.0),
            "B1": ("contact", 1.4, 2.4),
            "B2": ("push", 1.0, 1.5),
            "C1": ("push", 4.2, 5.0),
            "C2": ("push", 4.2, 5.2),
            "D1": ("release", 0.5, 1.5),
            "D2": ("release", 0.0, 1.0),
        }[key]
        if stage != phase:
            return 0.0
        t = self.elapsed(phase)
        if not start <= t < end:
            return 0.0
        ramp = min(0.25, (end - start) / 3)
        return float(np.clip((t - start) / ramp, 0, 1))

    def rate_scale(self):
        return (
            1 + 0.5 * self.pulse("push")
            if self.family == "C" and self.variation == "correction-2"
            else 1.0
        )

    def injected_reference_advance(self, nominal_step):
        """Execute the pace error beyond the unchanged ordinary reference cap.

        Scaling only its clock is ineffective whenever the ordinary lead cap is
        active. Accumulate the extra delivered progression separately; keep it
        fully applied until corrective control takes over, without a fall taper.
        """
        if self.family != "C" or self.variation != "correction-2" or self.pulse("push") <= 0:
            return None
        self.reference_advance_rad += (self.rate_scale() - 1) * max(0.0, nominal_step)
        return self.reference_advance_rad

    def material_gap_target(self, phase):
        """Declare separation from the moving material, not a stale world origin."""
        if self.family == "B" and self.variation == "correction-2" and self.pulse(phase) > 0:
            return NORMAL_RETREAT_M["B2"]
        if self.family == "D" and self.variation == "correction-1" and self.pulse(phase) > 0:
            return 0.008
        return None

    def holds_release_contact(self):
        return bool(
            self.family == "D" and self.variation == "correction-2" and self.pulse("release") > 0
        )

    def additional_contact_reserve_s(self):
        """Reserve the intentional pause and its complete corrective transition."""
        if self.family == "D" and self.variation == "correction-2":
            return 1.0 + self.corrective_transition_s
        return 0.0

    def position_return_active(self, phase):
        """The declared trajectory owns position error; force unloading stays active."""
        return self.pulse(phase) > 0 or (self.pending and not self.corrective_transition_complete)

    def target(
        self,
        position,
        rotation,
        phase,
        measured_position,
        measured_rotation=None,
        *,
        injected_pose=None,
    ):
        position, rotation = np.array(position, copy=True), np.array(rotation, copy=True)
        amount = self.pulse(phase)
        previous_role = self.role
        self.role = INJECTED if amount > 0 else CORRECTIVE if self.pending else EXPERT
        measured_rotation = rotation if measured_rotation is None else measured_rotation
        error = dict(
            position_m=float(np.linalg.norm(position - measured_position)),
            rotation_rad=float(Rotation.from_matrix(rotation @ measured_rotation.T).magnitude()),
        )
        # Retain the ordinary teacher reference before either injection or the
        # corrective blend. This evaluator trace never enters policy observations.
        self.pre_action_reference = dict(
            position_m=position.tolist(),
            quaternion_xyzw=Rotation.from_matrix(rotation).as_quat().tolist(),
            error=error,
        )
        if self.role == CORRECTIVE:
            if previous_role == INJECTED:
                # The disturbance stays fully applied until this handoff. Begin
                # corrective supervision from the actual residual, then execute
                # a smooth return to the ordinary task target through normal IK.
                self.corrective_origin = (
                    np.array(measured_position, copy=True),
                    np.array(measured_rotation, copy=True),
                )
                self.corrective_elapsed_s = 0.0
                self.corrective_transition_complete = False
                self.corrective_start_error = error
            self.corrective_latest_error = error
            if self.corrective_origin is not None:
                self.corrective_elapsed_s += self.dt
                moving_s = max(self.corrective_elapsed_s - self.corrective_brake_s, 0.0)
                blend = 0.5 - 0.5 * np.cos(
                    np.pi * min(moving_s / self.corrective_transition_s, 1.0)
                )
                self.corrective_transition_complete = blend >= 1.0
                origin_p, origin_r = self.corrective_origin
                position = origin_p + blend * (position - origin_p)
                delta = Rotation.from_matrix(rotation @ origin_r.T).as_rotvec()
                rotation = Rotation.from_rotvec(blend * delta).as_matrix() @ origin_r
        if amount > 0:
            self.pending = True
            key = self.family + self.variation[-1]
            if self.approach_disturbance is not None:
                d = self.approach_disturbance
                if d["kind"] == "pause":
                    if self.pause_pose is None:
                        self.pause_pose = (
                            np.array(measured_position, copy=True),
                            np.array(measured_rotation, copy=True),
                        )
                    position, rotation = (v.copy() for v in self.pause_pose)
                else:
                    position += rotation @ (amount * np.asarray(d["translation_m"]))
                    rotation = (
                        rotation
                        @ Rotation.from_rotvec(amount * np.asarray(d["rotation_rad"])).as_matrix()
                    )
            elif key == "A1":
                position += rotation[:, 1] * (0.02 * amount)
                rotation = (
                    rotation @ Rotation.from_rotvec([0, 0, np.deg2rad(5) * amount]).as_matrix()
                )
            elif key in ("A2", "B1", "C1"):
                position -= rotation[:, 0] * (NORMAL_RETREAT_M[key] * amount)
            elif key in ("B2", "D1", "D2"):
                if injected_pose is None:
                    kind = "material-relative" if key == "D2" else "material-gap"
                    raise ValueError(f"{key} requires an executed {kind} pose")
                injected_p, injected_r = injected_pose
                position += amount * (injected_p - position)
                delta = Rotation.from_matrix(injected_r @ rotation.T).as_rotvec()
                rotation = Rotation.from_rotvec(amount * delta).as_matrix() @ rotation
            elif key == "C2":
                if injected_pose is None:
                    raise ValueError("C2 requires an executed advanced reference pose")
                position, rotation = (np.array(v, copy=True) for v in injected_pose)
        self.counts[phase] = self.counts.get(phase, 0) + 1
        self.injected_commands += int(self.role == INJECTED)
        self.corrective_commands += int(self.role == CORRECTIVE)
        return position, rotation

    def supported_pace_exit(self, rows):
        """Handoff an executed pace pulse only with actual sustained loaded contact."""
        if not (
            self.family == "C"
            and self.variation == "correction-2"
            and self.role == INJECTED
            and self.injected_commands > 0
            and self.pulse("push") == 0
            and self.elapsed("push") >= 5.2
            and not self.supported_hold_handoffs
            and rows
        ):
            return False
        cutoff = rows[-1]["time"] - self.setup.sustain_s - 1e-9
        window = [row for row in rows if row["time"] >= cutoff]
        return bool(
            window[-1]["time"] - window[0]["time"] >= self.setup.sustain_s - 1e-9
            and all(row["valid"] and row["physical_safe"] for row in window)
        )

    def needs_supported_hold_handoff(self, row):
        """Hand sustained support to force holding; first impact is insufficient."""
        return bool(
            self.family == "C"
            and self.variation in ("correction-1", "correction-2")
            and self.role == CORRECTIVE
            and self.injected_commands > 0
            and self.recovered_samples * self.dt >= self.setup.sustain_s
            and not self.supported_hold_handoffs
            and row["phase"] == "push"
            and row["valid"]
            and row["physical_safe"]
        )

    def accept_supported_hold_handoff(self):
        """Runtime has reconstructed the achieved contact goal for force holding."""
        self.corrective_origin = None
        self.corrective_transition_complete = True
        self.supported_hold_handoffs += 1
        # Actual sustained loaded support must still complete this correction.

    def observe(self, row):
        row["action_role"] = self.role
        self.unloaded_s = 0.0 if row["loaded"] else self.unloaded_s + self.dt
        row["recovery_valid"] = bool(
            self.role == CORRECTIVE
            and self.pending
            and self.had_contact
            and not row["loaded"]
            and self.recontact_supported(row, self.unloaded_s)
        )
        if row["recovery_valid"]:
            row["phase_valid"] = True
        self.had_contact |= row["valid"]
        if self.pre_action_reference is not None:
            row["correction_reference"] = self.pre_action_reference
        if self.role == INJECTED:
            self.recovered_samples = 0
        elif self.pending:
            supported = (
                row["valid"]
                if self.family in ("A", "B", "C")
                else (not row["loaded"] and row["separation_m"] >= 0.01)
            )
            self.recovered_samples = self.recovered_samples + 1 if supported else 0
            if self.recovered_samples * self.dt >= self.setup.sustain_s:
                self.pending = False

    def permit_recontact(self, row, missing_s):
        """Only a named recoverable short deviation; never clear a physical stop/failure."""
        self.contact_loss_detected = True
        return self.recontact_supported(row, missing_s)

    def recontact_supported(self, row, missing_s):
        """Physical envelope for named short recovery, independent of action role."""
        return bool(
            self.family in ("B", "C")
            and self.variation.startswith("correction-")
            and missing_s <= 1.0
            and row["physical_safe"]
            and row["footprint_inside"]
            and row["material_error"] <= self.setup.position_tolerance
            and row["material_orientation_error"] <= self.setup.orientation_tolerance
            and row["force"] < self.setup.soft_force_n
        )

    def summary(self):
        return dict(
            injected_commands=self.injected_commands,
            corrective_commands=self.corrective_commands,
            pending_correction=self.pending,
            full_task_contact_loss=self.contact_loss_detected,
            approach_disturbance=self.approach_disturbance,
            corrective_start_error=self.corrective_start_error,
            corrective_latest_error=self.corrective_latest_error,
            injected_reference_advance_rad=self.reference_advance_rad,
            corrective_transition_s=self.corrective_transition_s,
            corrective_brake_s=self.corrective_brake_s,
            supported_hold_handoffs=self.supported_hold_handoffs,
            transfer="full disturbance until handoff; measured-origin cosine correction",
            action_roles={
                "expert": EXPERT,
                "injected_excluded": INJECTED,
                "corrective": CORRECTIVE,
            },
        )
