"""Optional observation of native registration; no decisions or truth inputs."""

import inspect
import json
from copy import deepcopy
from pathlib import Path

import numpy as np


def trace_value(value):
    if isinstance(value, dict):
        return {str(k): trace_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [trace_value(v) for v in value]
    if isinstance(value, np.generic):
        return trace_value(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


class RegistrationTrace:
    """Copy intermediate native values before later graph/map updates can alter them."""

    def __init__(self, pipeline, path):
        self.frame = None
        self.path = Path(path)
        register = pipeline.frontend.register
        native_register = register.register
        signature = inspect.signature(native_register)

        def observe_register(*args, **kwargs):
            call = signature.bind(*args, **kwargs)
            call.apply_defaults()
            values = call.arguments
            obj = self.object(values["obj_id"])
            obj["registration_input"] = deepcopy(
                {
                    k: values[k]
                    for k in ("src_pcd", "tgt_pcd", "sigma_tgt", "init_pose", "prev_T", "mode")
                }
            )
            result = native_register(*args, **kwargs)
            obj["registration_pose"] = result[0].copy()
            obj["registration_stats"] = deepcopy(result[1])
            return result

        register.register = observe_register
        native_refine = register._maybe_refine_with_sdf
        refine_signature = inspect.signature(native_refine)

        def observe_refine(*args, **kwargs):
            call = refine_signature.bind(*args, **kwargs)
            call.apply_defaults()
            values = call.arguments
            before = values["T_seed"].copy()
            result = native_refine(*args, **kwargs)
            self.object(values["obj_id"])["sdf_refinement"] = dict(
                before=before, after=result[0].copy(), info=deepcopy(result[1])
            )
            return result

        register._maybe_refine_with_sdf = observe_refine
        native_graph = pipeline.kf_graph.update

        def observe_graph(keyframes):
            inputs = [
                dict(object_id=k.obj_id, keyframe_index=k.kf_idx, pose=k.pose.copy())
                for k in keyframes
            ]
            result = native_graph(keyframes)
            self.frame["graph_updates"].append(
                dict(
                    inputs=inputs,
                    optimized_poses=[
                        dict(object_id=key[0], keyframe_index=key[1], pose=pose.copy())
                        for key, pose in result[0].items()
                    ],
                    landmarks=[
                        dict(object_id=key, count=len(value[0])) for key, value in result[1].items()
                    ],
                )
            )
            return result

        pipeline.kf_graph.update = observe_graph

    def object(self, object_id):
        return self.frame["objects"].setdefault(int(object_id), dict(object_id=int(object_id)))

    def begin(self, frame, capture_s, native_index, objects):
        self.frame = dict(
            frame=int(frame),
            capture_s=float(capture_s),
            native_index=native_index,
            objects={},
            graph_updates=[],
        )
        for obj in objects:
            self.object(obj.id).update(previous_pose=obj.pose.copy(), lost_before=bool(obj.lost))

    def frontend(self, result, objects):
        for obj in objects:
            self.object(obj.id).update(
                frontend_pose=deepcopy(result.obj_poses.get(obj.id)),
                frontend_lost=bool(obj.lost),
                frontend_source_points=deepcopy(result.valid_key_points.get(obj.id)),
                frontend_current_points=deepcopy(result.valid_curr_3d.get(obj.id)),
                frontend_track_indices=deepcopy(result.valid_indices.get(obj.id)),
                frontend_stats=deepcopy(result.reg_stats.get(obj.id)),
                extraction_stats=deepcopy(result.valid_stats.get(obj.id)),
                mean_residual=result.mean_residuals.get(obj.id),
                tracks=deepcopy(result.tracks[result.valid_indices[obj.id]])
                if obj.id in result.valid_indices
                else None,
            )

    def finish(self, objects):
        for obj in objects:
            self.object(obj.id).update(published_pose=obj.pose.copy(), lost=bool(obj.lost))
        record = dict(self.frame, objects=list(self.frame["objects"].values()))
        with self.path.open("a") as stream:
            stream.write(json.dumps(trace_value(record), allow_nan=False) + "\n")
