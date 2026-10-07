"""Diagnostic native allocation inventory; no inference decisions or truth inputs."""

import numpy as np


def tensor_inventory(value):
    """Count unique CUDA storages as well as logical tensor sizes (views can alias)."""
    import torch

    tensors, storages = [], {}

    def visit(item, path):
        if torch.is_tensor(item):
            tensors.append(
                dict(
                    path=path,
                    shape=list(item.shape),
                    bytes=item.numel() * item.element_size(),
                    device=str(item.device),
                )
            )
            if item.is_cuda:
                storage = item.untyped_storage()
                storages[storage.data_ptr()] = storage.nbytes()
        elif isinstance(item, dict):
            for key, child in item.items():
                visit(child, f"{path}.{key}")
        elif isinstance(item, (list, tuple)):
            for key, child in enumerate(item):
                visit(child, f"{path}.{key}")

    visit(value, "root")
    return dict(cuda_storage_bytes=sum(storages.values()), tensors=tensors)


def resource_snapshot(pipeline):
    """Inventory tracker, retained keyframes/graph and native TSDF across all objects."""
    import torch

    tracker, manager = pipeline.frontend.tracker, pipeline.kf_manager
    objects, masks = [], []
    for obj in pipeline.objects:
        kfs = manager.keyframes.get(obj.id, [])
        pending = manager.pending_keyframes.get(obj.id, [])
        all_kfs = kfs + [item["keyframe"] for item in pending]
        masks.extend(kf.frame.mask for kf in all_kfs)
        volume = obj.sdf_volume
        voxels = 0 if volume is None else int(np.prod(volume._vol_dim))
        optimizer = (
            None if pipeline.kf_graph is None else pipeline.kf_graph._optimizers.get(obj.id)
        )
        objects.append(
            dict(
                object_id=int(obj.id),
                active_references=len(pipeline.track_table.obj2track_map.get(obj.id, [])),
                historical_references=len(obj.key_points),
                keyframes=len(kfs),
                pending_keyframes=len(pending),
                pending_references=len(manager.pending_track_ids.get(obj.id, [])),
                dense_points=sum(len(kf.dense_pts) for kf in all_kfs if kf.dense_pts is not None),
                graph_poses=0 if optimizer is None else optimizer.get_num_poses(),
                graph_landmarks=0 if optimizer is None else len(optimizer._inserted_landmarks),
                graph_factors=0 if optimizer is None else optimizer._graph.size(),
                tsdf_voxels=voxels,
                tsdf_cuda_bytes=12 * voxels,
                tsdf_bounds=None if volume is None else volume._vol_bnds.tolist(),
            )
        )
    masks.extend(frame.mask for frame in pipeline.hist_frames)
    return dict(
        objects=objects,
        tracker_queries=0 if tracker.query_points is None else len(tracker.query_points),
        tapir_refinement_resolutions=(
            None if tracker.query_features is None else tracker.query_features.resolutions
        ),
        tapir_causal_levels=0 if tracker._causal_state is None else len(tracker._causal_state),
        query_points=tensor_inventory(tracker.query_points),
        query_features=tensor_inventory(tracker.query_features),
        causal_state=tensor_inventory(tracker._causal_state),
        retained_frame_masks=tensor_inventory(masks),
        torch_allocated_bytes=torch.cuda.memory_allocated(),
        torch_reserved_bytes=torch.cuda.memory_reserved(),
        torch_peak_bytes=torch.cuda.max_memory_allocated(),
        gpu_free_bytes=torch.cuda.mem_get_info()[0],
    )
