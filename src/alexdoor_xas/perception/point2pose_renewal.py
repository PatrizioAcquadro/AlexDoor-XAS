"""Bounded active TAPIR references with stable historical map/graph IDs."""

from collections import defaultdict

import numpy as np

# Measured complete baselines use at most four native 30-point batches per object.
ACTIVE_REFERENCE_BUDGET = 120


def published_support(obj, result, threshold):
    """Recompute support on the actual post-graph pose AND current landmarks, without GT."""
    ids = np.asarray(result.valid_indices.get(obj.id, []), dtype=int)
    current = np.asarray(result.valid_curr_3d.get(obj.id, []), dtype=float).reshape(-1, 3)
    rows = obj.track_idx_2_obj_idx[ids]
    if len(ids) != len(current) or np.any(rows < 0):
        raise ValueError("invalid_published_correspondence_ids")
    source = obj.key_points[rows]
    residuals = np.linalg.norm(source @ obj.pose[:3, :3].T + obj.pose[:3, 3] - current, axis=1)
    return ids, residuals, np.isfinite(residuals) & (residuals <= threshold)


def compact_tracker(tracker, keep):
    """Copy retained point rows into independent storages; release discarded CUDA state."""
    import torch

    n = len(tracker.query_points)
    index = torch.as_tensor(keep, dtype=torch.long, device=tracker.query_points.device)
    features = tracker.query_features
    tracker.query_points = tracker.query_points.index_select(0, index)
    tracker.query_features = features._replace(
        lowres=[value.index_select(1, index) for value in features.lowres],
        hires=[value.index_select(1, index) for value in features.hires],
    )
    state = []
    for level in tracker._causal_state:
        retained = {}
        for key, value in level.items():
            if torch.is_tensor(value):
                if value.ndim < 2 or value.shape[1] != n:
                    raise ValueError("unexpected_tapir_point_axis")
                value = value.index_select(1, index)
            retained[key] = value
        state.append(retained)
    tracker._causal_state = state


def geometric_guard(points):
    """Protect the planar hull and one representative of each occupied spatial cell."""
    from scipy.spatial import ConvexHull, QhullError

    points = np.asarray(points, dtype=float)
    if len(points) < 4 or not np.isfinite(points).all():
        return set(range(len(points)))
    centered = points - points.mean(0)
    _, _, axes = np.linalg.svd(centered, full_matrices=False)
    xy = centered @ axes[:2].T
    try:
        protected = set(ConvexHull(xy).vertices.tolist())
    except QhullError:
        return set(range(len(points)))
    span = np.ptp(xy, axis=0)
    cells = np.minimum(2, ((xy - xy.min(0)) / np.maximum(span, 1e-12) * 3).astype(int))
    for cell in np.unique(cells, axis=0):
        members = np.flatnonzero((cells == cell).all(1))
        if len(members) == 1:
            protected.add(int(members[0]))
    return protected


class BoundedRenewal:
    """Keep original sampling/promotions/graph; bound admission and retire confirmed substitutes."""

    def __init__(self, pipeline, budget=ACTIVE_REFERENCE_BUDGET):
        self.pipeline, self.budget = pipeline, int(budget)
        self.active_ids = np.empty(0, dtype=int)
        self.next_id = 0
        self.utility = defaultdict(lambda: dict(bad=0, good=0))
        self.spent = set()
        self.events = []
        tracker, manager = pipeline.frontend.tracker, pipeline.kf_manager
        native_add, native_track, native_sample = (
            tracker.add_query_points,
            tracker.track_once,
            manager.sampler.sample,
        )

        def add(frame, points):
            slots = native_add(frame, points)
            ids = np.arange(self.next_id, self.next_id + len(slots))
            self.next_id += len(slots)
            self.active_ids = np.concatenate((self.active_ids, ids))
            return ids

        def track(frame):
            compact = native_track(frame)
            outputs = []
            for values, fill in zip(compact, (-1, 1, False), strict=True):
                full = np.full((self.next_id, *values.shape[1:]), fill, dtype=values.dtype)
                full[self.active_ids] = values
                outputs.append(full)
            return tuple(outputs)

        def sample(context, obj_id):
            # Defer a whole native batch. Never alter the sampled point order/selection.
            count = len(pipeline.track_table.obj2track_map.get(obj_id, []))
            if count + manager.sampler.num_points > self.budget:
                self.events.append(dict(object_id=obj_id, kind="budget_deferred", count=count))
                return np.empty((0, 2), dtype=float)
            return native_sample(context, obj_id)

        tracker.add_query_points, tracker.track_once, manager.sampler.sample = add, track, sample

    def after_frame(self, frame, result):
        """Replace after published-pose confirmation; missing evidence never ages points."""
        if result is None:
            return
        pipeline, retired = self.pipeline, []
        table, manager = pipeline.track_table, pipeline.kf_manager
        threshold = pipeline.frontend.register._inlier_thres
        for obj in pipeline.objects:
            if obj.lost:
                continue
            ids, _, support = published_support(obj, result, threshold)
            # Only a supported published pose may establish usefulness or replacement credit.
            if support.sum() < pipeline.frontend.register._min_inliers:
                continue
            inliers = set(ids[support].tolist())
            active = table.obj2track_map.get(obj.id, np.empty(0, int))
            rows = obj.track_idx_2_obj_idx[active]
            eligible = set()
            mask = frame.mask[obj.id, 0]
            if hasattr(mask, "detach"):
                mask = mask.detach().cpu().numpy()
            for tid, row in zip(active, rows, strict=True):
                uv = table.track_2d[tid]
                if not (table.visible[tid] and table.valid[tid] and np.isfinite(uv).all()):
                    continue  # Freeze occlusion/invalid-depth intervals, even if previously poor.
                x, y = np.rint(uv).astype(int)
                if not (0 <= x < mask.shape[1] and 0 <= y < mask.shape[0] and mask[y, x] > 0):
                    continue
                if table.uncertainty[tid] >= manager.pending_uncer_thres:
                    continue
                eligible.add(int(tid))
                state = self.utility[int(tid)]
                if tid in inliers and obj.valid[row]:
                    state.update(bad=0, good=state["good"] + 1)
                else:
                    state.update(bad=state["bad"] + 1, good=0)
            replacements = [
                int(tid)
                for tid, row in zip(active, rows, strict=True)
                if obj.key_point_frames[row] > 0
                and tid in inliers
                and tid not in self.spent
                and self.utility[int(tid)]["good"] >= manager.pending_promote_streak
            ]
            poor = sorted(
                [
                    int(tid)
                    for tid in active
                    if tid in eligible
                    and self.utility[int(tid)]["bad"] > manager.pending_ttl
                    and (obj.id, int(tid)) not in manager.pending_meta
                ],
                key=lambda tid: (-self.utility[tid]["bad"], tid),
            )
            remaining = active.tolist()
            object_retired = 0
            for tid in poor:
                if not replacements or object_retired >= manager.pending_promote_streak:
                    break
                # Removing an interior/redundant point preserves hull and cell occupancy.
                confirmed = [t for t in remaining if obj.valid[obj.track_idx_2_obj_idx[t]]]
                protected = {
                    confirmed[i]
                    for i in geometric_guard(obj.key_points[obj.track_idx_2_obj_idx[confirmed]])
                }
                if tid in protected:
                    continue
                candidates = [t for t in replacements if t != tid]
                if not candidates:
                    continue
                row = obj.track_idx_2_obj_idx[tid]
                replacement = min(
                    candidates,
                    key=lambda t: np.linalg.norm(
                        obj.key_points[obj.track_idx_2_obj_idx[t]] - obj.key_points[row]
                    ),
                )
                self.spent.add(replacement)
                replacements.remove(replacement)
                remaining.remove(tid)
                retired.append(tid)
                object_retired += 1
                manager.promoted_meta.pop((obj.id, tid), None)
                self.events.append(
                    dict(
                        object_id=obj.id,
                        kind="retired",
                        track_id=tid,
                        replacement_id=replacement,
                        visible_bad_observations=self.utility[tid]["bad"],
                    )
                )
            table.obj2track_map[obj.id] = np.asarray(remaining, dtype=int)
        if retired:
            keep = np.flatnonzero(~np.isin(self.active_ids, retired))
            compact_tracker(pipeline.frontend.tracker, keep)
            self.active_ids = self.active_ids[keep]
            for tid in retired:
                self.utility.pop(tid, None)
            # Historical CPU arrays, IDs, keyframes, map validity and graph stay intact.
