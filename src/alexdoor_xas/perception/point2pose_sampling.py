"""Opt-in renewal selection repair using the pinned sampler's existing depth gate."""


def install_reference_depth_edge_filter(sampler, is_initialized):
    """Filter selected renewal points, preserving the original seed and order.

    The balanced sampler bypasses its inherited depth-discontinuity gate. Apply
    only that gate, before bounded admission; do not refill, re-rank, change the
    mask, or activate the other inherited candidate filters.
    """
    if not sampler.sample_filter_enable or not sampler.sample_reject_depth_edge:
        raise ValueError("reference_depth_edge_filter_requires_enabled_native_config")
    native_sample = sampler.sample

    def sample(context, obj_id):
        points = native_sample(context, obj_id)
        if not is_initialized() or not len(points):
            return points
        depth = sampler._frame_depth_meters(context.frame)
        if depth is None:
            raise ValueError("reference_depth_edge_filter_requires_measured_depth")
        keep = sampler._depth_edge_keep_mask(
            depth_m=depth,
            pts_global=points,
            patch_radius=sampler.sample_depth_edge_window,
            max_span_m=sampler.sample_depth_edge_thres,
        )
        sampler.reference_depth_edge_filter_event = dict(
            native_index=context.frame.id,
            object_id=obj_id,
            selected_points=points.copy(),
            keep=keep.copy(),
            radius_px=sampler.sample_depth_edge_window,
            max_span_m=sampler.sample_depth_edge_thres,
        )
        return points[keep]

    sampler.sample = sample
