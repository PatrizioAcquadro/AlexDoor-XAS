"""Bounded executed disturbances and measured-response bookkeeping for short tasks."""

import numpy as np
from scipy.spatial.transform import Rotation

# These diagnostic annotations never enter observation streams.
EXPERT, INJECTED, CORRECTIVE = 0, 1, 2


class CollectionCorrection:
    def __init__(self, family, variation, setup, dt):
        if family not in ("A", "B", "C", "D"):
            raise ValueError("Unknown phase family")
        self.family, self.variation, self.setup, self.dt = family, variation, setup, dt
        self.counts = {}
        self.role = EXPERT
        self.pending = False
        self.injected_commands = self.corrective_commands = 0
        self.recovered_samples = 0
        self.release_origin = None
        self.contact_loss_detected = False

    def elapsed(self, phase):
        return self.counts.get(phase, 0) * self.dt

    def pulse(self, phase):
        if not self.variation.startswith("correction-"):
            return 0.0
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
        return float(np.clip(min((t - start) / ramp, (end - t) / ramp), 0, 1))

    def rate_scale(self):
        return (
            1 + 0.5 * self.pulse("push")
            if self.family == "C" and self.variation == "correction-2"
            else 1.0
        )

    def target(self, position, rotation, phase, measured_position):
        position, rotation = np.array(position, copy=True), np.array(rotation, copy=True)
        if phase == "release" and self.release_origin is None:
            self.release_origin = np.array(measured_position, copy=True)
        amount = self.pulse(phase)
        self.role = INJECTED if amount > 0 else CORRECTIVE if self.pending else EXPERT
        if amount > 0:
            self.pending = True
            key = self.family + self.variation[-1]
            if key == "A1":
                position += rotation[:, 1] * (0.02 * amount)
                rotation = (
                    rotation @ Rotation.from_rotvec([0, 0, np.deg2rad(5) * amount]).as_matrix()
                )
            elif key in ("A2", "B1", "B2", "C1"):
                distance = {"A2": 0.015, "B1": 0.003, "B2": 0.0015, "C1": 0.001}[key]
                position -= rotation[:, 0] * (distance * amount)
            elif key == "D1":
                normal = float((position - self.release_origin) @ rotation[:, 0])
                position += rotation[:, 0] * max(0.0, -0.008 - normal) * amount
            elif key == "D2":
                position += (self.release_origin - position) * amount
            # C2 perturbs the actual reference advancement, not recorded states.
        self.counts[phase] = self.counts.get(phase, 0) + 1
        self.injected_commands += int(self.role == INJECTED)
        self.corrective_commands += int(self.role == CORRECTIVE)
        return position, rotation

    def observe(self, row):
        row["action_role"] = self.role
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
            action_roles={
                "expert": EXPERT,
                "injected_excluded": INJECTED,
                "corrective": CORRECTIVE,
            },
        )
