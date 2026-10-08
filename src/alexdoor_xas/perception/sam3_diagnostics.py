"""Observe native detection association and actual reconditioning, without GT."""


class ReconditioningTrace:
    def __init__(self, model):
        self.current = {}
        self.in_reconditioning = False
        plan_native = model.run_tracker_update_planning_phase
        recondition_native = model._recondition_masklets
        mask_native = model.tracker.add_new_mask

        def planning(*args, **kwargs):
            plan, metadata = plan_native(*args, **kwargs)
            self.current.update(
                new_detection_ids=[int(i) for i in plan["new_det_obj_ids"]],
                matched_detection_ids={
                    str(k): int(v) for k, v in plan["trk_id_to_max_iou_high_conf_det"].items()
                },
                output_detection_override_ids=sorted(int(i) for i in plan["reconditioned_obj_ids"]),
            )
            return plan, metadata

        def reconditioning(*args, **kwargs):
            self.current["called"] = True
            self.in_reconditioning = True
            try:
                return recondition_native(*args, **kwargs)
            finally:
                self.in_reconditioning = False

        def add_mask(*args, **kwargs):
            result = mask_native(*args, **kwargs)
            if self.in_reconditioning:
                self.current["applied_ids"].append(int(kwargs["obj_id"]))
            return result

        model.run_tracker_update_planning_phase = planning
        model._recondition_masklets = reconditioning
        model.tracker.add_new_mask = add_mask

    def begin(self):
        self.current = dict(called=False, applied_ids=[])


def set_detection_reconditioning(model, enabled):
    """Isolate the native periodic mechanism, retaining detection/other heuristics."""
    if not enabled:
        # The pinned SAM3 recipe have no bbox-triggered reconditioning. Do not
        # silently disable a second mechanism if another recipe enables it.
        if model.reconstruction_bbox_iou_thresh > 0:
            raise ValueError("unsupported_additional_reconditioning_trigger")
        model.recondition_every_nth_frame = 0
