"""Essential budget, identity, occlusion and actual CUDA storage regressions."""

from collections import namedtuple
from types import SimpleNamespace as NS

import numpy as np
import pytest

from alexdoor_xas.perception.point2pose.renewal import (
    BoundedRenewal,
    compact_tracker,
    geometric_guard,
    published_support,
)


def test_published_support_uses_post_graph_pose_and_landmarks():
    obj = NS(id=0, pose=np.eye(4), key_points=np.zeros((6, 3)), track_idx_2_obj_idx=np.arange(6))
    result = NS(valid_indices={0: np.arange(6)}, valid_curr_3d={0: np.zeros((6, 3))})
    assert published_support(obj, result, 0.004)[2].all()
    obj.pose[0, 3] = 0.01
    assert not published_support(obj, result, 0.004)[2].any()
    obj.key_points[:, 0] = -0.01
    assert published_support(obj, result, 0.004)[2].all()


def fixture_pipeline(*, allow_partial_batch=False):
    torch = pytest.importorskip("torch")
    points = np.array(
        [
            [0, 0, 1],
            [1, 0, 1],
            [1, 1, 1],
            [0, 1, 1],
            [0.5, 0.5, 1],
            [0.5, 0.5, 1],
            [0.2, 0.2, 1],
            [0.8, 0.2, 1],
            [0.2, 0.8, 1],
            [0.8, 0.8, 1],
        ],
        float,
    )
    features = namedtuple("Features", "lowres hires resolutions")
    tracker = NS(
        query_points=torch.zeros((10, 3)),
        query_features=features([torch.ones((1, 10, 2))], [torch.ones((1, 10, 1))], [(1, 1)]),
        _causal_state=[{"state": torch.ones((1, 10, 2, 3))}],
    )

    def add(frame, p):
        n = len(tracker.query_points)
        tracker.query_points = torch.cat((tracker.query_points, torch.zeros((len(p), 3))))
        return np.arange(n, n + len(p))

    tracker.add_query_points = add
    tracker.track_once = lambda frame: (
        np.arange(len(tracker.query_points) * 2).reshape(-1, 2),
        np.zeros(len(tracker.query_points)),
        np.ones(len(tracker.query_points), bool),
    )
    table = NS(
        obj2track_map={0: np.arange(10)},
        track_2d=np.ones((10, 2)) * 2,
        visible=np.ones(10, bool),
        valid=np.ones(10, bool),
        uncertainty=np.zeros(10),
    )
    obj = NS(
        id=0,
        lost=False,
        pose=np.eye(4),
        key_points=points,
        valid=np.ones(10, bool),
        track_idx_2_obj_idx=np.arange(10),
        key_point_frames=np.array([0] * 5 + [1] * 5),
    )
    sampler = NS(num_points=3, sample=lambda ctx, oid: np.array([[1, 2], [2, 3], [3, 4]], float))
    manager = NS(
        sampler=sampler,
        pending_uncer_thres=0.3,
        pending_promote_streak=3,
        pending_ttl=15,
        pending_meta={},
        promoted_meta={},
    )
    pipeline = NS(
        frontend=NS(tracker=tracker, register=NS(_inlier_thres=0.004, _min_inliers=5)),
        kf_manager=manager,
        track_table=table,
        objects=[obj],
    )
    renewal = BoundedRenewal(pipeline, budget=12, allow_partial_batch=allow_partial_batch)
    renewal.active_ids = np.arange(10)
    renewal.next_id = 10
    frame = NS(mask=np.ones((1, 1, 5, 5)), id=10)
    current = points.copy()
    current[4, 0] += 0.02
    result = NS(valid_indices={0: np.arange(10)}, valid_curr_3d={0: current})
    return pipeline, renewal, frame, result


def test_batch_budget_defers_without_resampling_or_id_reuse():
    p, r, f, _ = fixture_pipeline()
    assert p.kf_manager.sampler.sample(None, 0).shape == (0, 2)
    p.track_table.obj2track_map[0] = np.arange(8)
    assert len(p.kf_manager.sampler.sample(None, 0)) == 3
    compact_tracker(p.frontend.tracker, np.arange(8))
    r.active_ids = np.arange(8)
    ids = p.frontend.tracker.add_query_points(f, np.zeros((2, 2)))
    assert ids.tolist() == [10, 11] and r.active_ids.tolist() == list(range(8)) + [10, 11]
    tracks, unc, visible = p.frontend.tracker.track_once(f)
    assert tracks.shape == (12, 2) and (tracks[8:10] == -1).all()
    assert not visible[8:10].any() and (unc[8:10] == 1).all()


def test_partial_batch_preserves_native_prefix_and_does_not_exceed_capacity():
    p, r, _, _ = fixture_pipeline(allow_partial_batch=True)
    np.testing.assert_array_equal(p.kf_manager.sampler.sample(None, 0), [[1, 2], [2, 3]])
    assert r.events[-1]["admitted"] == 2
    p.track_table.obj2track_map[0] = np.arange(11)
    np.testing.assert_array_equal(p.kf_manager.sampler.sample(None, 0), [[1, 2]])
    p.track_table.obj2track_map[0] = np.arange(12)
    assert p.kf_manager.sampler.sample(None, 0).shape == (0, 2)


def test_retirement_requires_confirmed_substitute_and_freezes_occlusion_loss():
    p, r, f, result = fixture_pipeline()
    obj = p.objects[0]
    r.utility[4]["bad"] = 15
    p.track_table.visible[4] = False
    for _ in range(5):
        r.after_frame(f, result)
    assert r.utility[4]["bad"] == 15 and len(r.active_ids) == 10
    p.track_table.visible[4] = True
    obj.lost = True
    r.after_frame(f, result)
    assert r.utility[4]["bad"] == 15 and len(r.active_ids) == 10
    obj.lost = False
    # Suppress fresh confirmation: poor visibility alone cannot release old references.
    obj.key_point_frames[:] = 0
    r.after_frame(f, result)
    assert len(r.active_ids) == 10
    obj.key_point_frames[5:] = 1
    r.after_frame(f, result)
    assert 4 not in r.active_ids and len(r.active_ids) == 9
    assert len(obj.key_points) == 10 and obj.valid[4] and obj.track_idx_2_obj_idx[4] == 4
    assert set([0, 1, 2, 3]).issubset(r.active_ids)
    assert p.frontend.tracker.query_points.shape[0] == 9
    assert r.events[-1]["replacement_id"] == 5


def test_geometric_extremes_and_sparse_cells_remain_protected():
    points = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0.5, 0.5, 0], [0.5, 0.5, 0]])
    guard = geometric_guard(points)
    assert set(range(4)).issubset(guard) and 4 not in guard and 5 not in guard
