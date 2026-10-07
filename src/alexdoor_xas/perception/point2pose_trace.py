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
        self.pipeline = pipeline
        self.path = Path(path)
        segmenter = getattr(pipeline.frontend, "segmenter", None)
        if segmenter is not None:
            native_prompt = segmenter.predictor.add_new_prompt

            def observe_prompt(*args, **kwargs):
                self.frame.setdefault("sam2_prompts", []).append(
                    deepcopy({k: kwargs[k] for k in ("frame_idx", "obj_id", "points", "labels")})
                )
                return native_prompt(*args, **kwargs)

            segmenter.predictor.add_new_prompt = observe_prompt
        manager = getattr(pipeline, "kf_manager", None)
        if manager is not None:
            native_promote = manager._pending_try_promote

            def observe_promote(key, meta, obj):
                eligible = meta.get("good", 0) >= manager.pending_promote_streak
                record = None
                if eligible:
                    geom_ok, geometry = manager._pending_obs_geom_ok(meta)
                    record = dict(
                        object_id=key[0],
                        track_id=key[1],
                        geometric_check_enabled=manager.pending_use_geom_check,
                        geometric_check_passed=geom_ok,
                        geometry=deepcopy(geometry),
                        metadata=deepcopy(meta),
                        point_before=obj.key_points[meta["obj_idx"]].copy(),
                    )
                result = native_promote(key, meta, obj)
                if record is not None:
                    record.update(
                        promoted=result, point_after=obj.key_points[meta["obj_idx"]].copy()
                    )
                    self.frame.setdefault("promotion_checks", []).append(record)
                return result

            manager._pending_try_promote = observe_promote
        register = pipeline.frontend.register
        native_register = register.register
        signature = inspect.signature(native_register)

        def observe_register(*args, **kwargs):
            call = signature.bind(*args, **kwargs)
            call.apply_defaults()
            values = call.arguments
            # Official SVDResidualOutlierRegister places metadata in **kwargs.
            values = dict(values, **values.get("kwargs", {}))
            obj = self.object(values.get("obj_id", 0))
            obj["registration_input"] = deepcopy(
                {
                    k: values.get(k)
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
        if pipeline.kf_graph is None:
            return
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
            extraction = result.valid_stats.get(obj.id, {})
            indices = np.asarray(extraction.get("extract_obj_idx", []), dtype=int)
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
                all_track_indices=indices.copy(),
                all_tracks=deepcopy(result.tracks[indices]) if len(indices) else None,
                all_uncertainties=deepcopy(result.uncertainties[indices]) if len(indices) else None,
                all_visibles=deepcopy(result.visibles[indices]) if len(indices) else None,
                all_current_points=deepcopy(result.track_3d[indices]) if len(indices) else None,
            )

    def finish(self, objects, *, masks=None, track_table=None, renewal=None):
        if masks is not None:
            name = f"references-{self.frame['frame']}.npz"
            np.savez_compressed(self.path.parent / name, masks=masks)
            self.frame["mask_file"] = name
        for obj in objects:
            self.object(obj.id).update(published_pose=obj.pose.copy(), lost=bool(obj.lost))
            if track_table is not None:
                indices = np.asarray(track_table.obj2track_map[obj.id], dtype=int)
                rows = obj.track_idx_2_obj_idx[indices]
                valid = (rows >= 0) & (rows < len(obj.key_points))
                indices, rows = indices[valid], rows[valid]
                self.object(obj.id).update(
                    map_track_indices=indices.copy(),
                    map_points=obj.key_points[rows].copy(),
                    map_valid=obj.valid[rows].copy(),
                    map_tracks=track_table.track_2d[indices].copy(),
                )
                selected = self.object(obj.id).get("frontend_track_indices")
                selected = np.asarray([] if selected is None else selected, dtype=int)
                self.object(obj.id)["published_source_points"] = obj.key_points[
                    obj.track_idx_2_obj_idx[selected]
                ].copy()
        if renewal is not None:
            self.frame["renewal_events"] = deepcopy(renewal.events)
            self.frame["active_query_ids"] = renewal.active_ids.copy()
            for obj in objects:
                self.object(obj.id)["historical_reference_count"] = len(obj.key_points)
        record = dict(self.frame, objects=list(self.frame["objects"].values()))
        with self.path.open("a") as stream:
            stream.write(json.dumps(trace_value(record), allow_nan=False) + "\n")
        if hasattr(self.pipeline, "track_table"):
            from alexdoor_xas.perception.point2pose_resources import resource_snapshot

            resources = dict(
                frame=self.frame["frame"],
                capture_s=self.frame["capture_s"],
                **resource_snapshot(self.pipeline),
            )
            with self.path.with_name("resources.jsonl").open("a") as stream:
                stream.write(json.dumps(resources, allow_nan=False) + "\n")
