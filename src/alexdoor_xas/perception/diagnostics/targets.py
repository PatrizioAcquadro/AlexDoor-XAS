"""Score fixed physical targets on saved outputs; truth never enters the coordinator."""

import numpy as np

from alexdoor_xas.perception.diagnostics.metrics import evaluate_bounds
from alexdoor_xas.perception.diagnostics.offline import finite_pose
from alexdoor_xas.perception.diagnostics.replay import pose_error, truth_panel


def evaluate_targets(h5, frames, targets, *, reference_row, seed_row, object_id=0):
    """Targets are world poses at reference_row, independently frozen by the evaluator.

    Re-express those same material targets at the actual P2P seed using evaluator
    motion. Do not replace them with the new geometric zone or align each output
    to truth. Missing and lost outputs retain their original scheduled denominator.
    """
    targets = [np.asarray(target, float) for target in targets]
    if not targets or any(not finite_pose(target) for target in targets):
        raise ValueError("invalid_evaluator_targets")
    series = [[] for _ in targets]
    if seed_row is None:
        for frame in frames:
            for rows in series:
                rows.append(dict(time_s=frame["time_s"], is_seed=False))
    else:
        seed_truth = truth_panel(h5, seed_row)
        reference_truth = truth_panel(h5, reference_row)
        at_seed = [seed_truth @ np.linalg.inv(reference_truth) @ target for target in targets]
        seed_camera = h5["observations/camera_world"][seed_row]
        for frame in frames:
            obj = next((o for o in frame["objects"] if o.get("object_id") == object_id), {})
            pose = obj.get("camera_from_map")
            valid = pose is not None and finite_pose(pose)
            world_motion = (
                h5["observations/camera_world"][frame["row"]]
                @ np.asarray(pose)
                @ np.linalg.inv(seed_camera)
                if valid
                else None
            )
            truth_motion = truth_panel(h5, frame["row"]) @ np.linalg.inv(seed_truth)
            for target, rows in zip(at_seed, series, strict=True):
                error = (
                    None
                    if world_motion is None
                    else pose_error(world_motion @ target, truth_motion @ target)
                )
                rows.append(
                    dict(
                        time_s=frame["time_s"],
                        is_seed=frame.get("is_seed", False),
                        native_pose_valid=valid,
                        native_tracking=obj.get("native_tracking", False),
                        native_lost=obj.get("native_lost", False),
                        integration_accepted=obj.get("integration_accepted", False),
                        capture_error=error,
                    )
                )
    return [
        dict(
            target_id=i,
            world_at_reference=target.tolist(),
            reference_row=reference_row,
            seed_row=seed_row,
            metrics=evaluate_bounds(rows),
        )
        for i, (target, rows) in enumerate(zip(targets, series, strict=True))
    ]
