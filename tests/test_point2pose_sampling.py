"""Essential opt-in renewal gate semantics; no model inference."""

from types import SimpleNamespace

import numpy as np

from alexdoor_xas.perception.point2pose_sampling import install_reference_depth_edge_filter


def test_depth_edge_gate_preserves_seed_order_and_never_restores_rejected_points():
    points = np.array([[20, 20], [8, 5], [12, 12]], dtype=np.int32)
    calls = []
    initialized = False

    def gate(**kwargs):
        calls.append(kwargs)
        return np.array([True, False, True]) if len(calls) == 1 else np.zeros(3, bool)

    depth = object()
    sampler = SimpleNamespace(
        sample=lambda context, obj_id: points,
        sample_filter_enable=True,
        sample_reject_depth_edge=True,
        sample_depth_edge_window=2,
        sample_depth_edge_thres=0.01,
        depth_fallback_to_unfiltered=True,
        _frame_depth_meters=lambda frame: depth,
        _depth_edge_keep_mask=gate,
    )
    context = SimpleNamespace(frame=SimpleNamespace(id=1868))
    install_reference_depth_edge_filter(sampler, lambda: initialized)
    assert sampler.sample(context, 0) is points  # Original initialization is exact.
    assert not calls
    initialized = True
    context.frame.id = 2133
    assert np.array_equal(sampler.sample(context, 0), points[[0, 2]])
    assert calls[0]["depth_m"] is depth and calls[0]["pts_global"] is points
    assert calls[0]["patch_radius"] == 2 and calls[0]["max_span_m"] == 0.01
    assert np.array_equal(sampler.reference_depth_edge_filter_event["keep"], [True, False, True])
    assert sampler.sample(context, 0).shape == (0, 2)
    assert np.array_equal(points, [[20, 20], [8, 5], [12, 12]])
