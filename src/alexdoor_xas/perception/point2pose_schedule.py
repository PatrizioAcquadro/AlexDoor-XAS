"""Diagnostic registration scheduling; retain every candidate's pixel tracker.

The selected object is the first eligible automatic observed candidate, as in
the serial evaluator. Selection does not establish leaf ownership. Deferred
objects have no current registration evidence and cannot renew their map.
"""

import numpy as np


class SelectedRegistration:
    def __init__(self, pipeline, *, audit_frames=9):
        if pipeline.use_key_frame_graph or pipeline.frontend.frame_reg_mode != "f2m":
            raise ValueError("selected_registration_requires_graph_off_f2m")
        self.pipeline = pipeline
        self.audit_frames = int(audit_frames)
        if self.audit_frames < 1:
            raise ValueError("invalid_registration_audit_period")
        self.selected = 0
        self.evaluated = set()
        self.expanded = False
        self.reason = "initialization"
        self.frame_id = 0
        native = pipeline.frontend._extract_valid_key_points_mask_remove

        def extract(obj, obj_idx, tracks, points, visible, valid, uncertainty, mask, **kwargs):
            # The selected object is first in the native ordered loop. Its
            # current refusal expands recovery in this same acquisition.
            if pipeline.objects[self.selected].lost:
                self.expanded, self.reason = True, "selected_support_lost"
            if obj.id == self.selected or self.expanded:
                self.evaluated.add(obj.id)
                return native(
                    obj, obj_idx, tracks, points, visible, valid, uncertainty, mask, **kwargs
                )
            # Native insufficient-pair handling freezes the pose, marks it lost
            # and suppresses keyframes/renewal. SAM2/TAPIR/table still advance.
            return (
                np.empty(0, dtype=int),
                np.empty((0, 3)),
                np.empty((0, 3)),
                visible,
                {"registration_deferred": True},
            )

        pipeline.frontend._extract_valid_key_points_mask_remove = extract

    def begin(self, frame_id):
        self.frame_id = int(frame_id)
        self.evaluated.clear()
        self.expanded = self.frame_id % self.audit_frames == 0
        self.reason = "periodic_audit" if self.expanded else "selected_only"
        if self.pipeline.objects and self.pipeline.objects[self.selected].lost:
            self.expanded, self.reason = True, "selected_support_lost"

    def metadata(self):
        return dict(
            selected_object_id=self.selected,
            audit_frames=self.audit_frames,
            evaluated_object_ids=sorted(self.evaluated),
            reason=self.reason,
            frame_id=self.frame_id,
        )
