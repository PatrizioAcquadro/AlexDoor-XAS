"""Diagnostic observation must preserve native decisions and intermediate values."""

import json
from types import SimpleNamespace

import numpy as np
import pytest

from alexdoor_xas.perception.point2pose_trace import RegistrationTrace


@pytest.mark.parametrize("with_graph", [True, False])
def test_trace_preserves_native_returns_and_copies_before_graph_changes(tmp_path, with_graph):
    class Register:
        def _maybe_refine_with_sdf(self, T_seed, obj_id):
            refined = T_seed.copy()
            refined[0, 3] += 0.02
            self.refined = (refined, dict(applied=True, seed_cost=np.inf))
            return self.refined

        def register(
            self,
            src_pcd,
            tgt_pcd,
            sigma_tgt=None,
            init_pose=None,
            prev_T=None,
            mode="f2m",
            obj_id=0,
        ):
            self.refine_return = self._maybe_refine_with_sdf(init_pose, obj_id)
            self.result = (
                init_pose,
                dict(inliers=np.ones(len(src_pcd), bool), residuals=np.zeros(len(src_pcd))),
            )
            return self.result

    class Graph:
        def update(self, keyframes):
            keyframes[0].pose[1, 3] = 0.03
            self.result = ({(0, 1): keyframes[0].pose}, {0: (np.zeros((2, 3)), np.arange(2))})
            return self.result

    register, graph = Register(), Graph()
    prompt_result = object()
    predictor = SimpleNamespace(add_new_prompt=lambda **kwargs: prompt_result)
    meta = dict(good=3, obj_idx=0, obs_obj=[[0, 0, 1], [0.01, 0, 1], [0.02, 0, 1]])

    def promote(key, meta, obj):
        obj.key_points[meta["obj_idx"], 0] = 0.01
        return True

    manager = SimpleNamespace(
        pending_promote_streak=3,
        pending_use_geom_check=False,
        _pending_obs_geom_ok=lambda meta: (False, dict(n_obs=3, spread=0.01)),
        _pending_try_promote=promote,
    )
    pipeline = SimpleNamespace(
        frontend=SimpleNamespace(register=register, segmenter=SimpleNamespace(predictor=predictor)),
        kf_graph=graph if with_graph else None,
        kf_manager=manager,
    )
    path = tmp_path / "registration.jsonl"
    trace = RegistrationTrace(pipeline, path)
    obj = SimpleNamespace(id=0, pose=np.eye(4), lost=False)
    trace.begin(248, 4.0, 0, [obj])
    prompts = np.array([[1, 2], [3, 4]])
    assert (
        predictor.add_new_prompt(frame_idx=0, obj_id=0, points=prompts, labels=np.ones(2))
        is prompt_result
    )
    prompts[:] = 9
    obj.key_points = np.array([[0.0, 0, 1]])
    assert manager._pending_try_promote((0, 0), meta, obj)
    meta["good"] = 0
    points = np.array([[0.0, 0, 1], [1, 0, 1], [0, 1, 1]])
    result = register.register(points, points, init_pose=obj.pose, prev_T=obj.pose)
    assert result is register.result and register.refine_return is register.refined
    frontend = SimpleNamespace(
        obj_poses={0: obj.pose},
        valid_key_points={0: points},
        valid_curr_3d={0: points},
        valid_indices={0: np.arange(3)},
        reg_stats={0: result[1]},
        valid_stats={0: dict(confirmed=3, extract_obj_idx=np.arange(3))},
        mean_residuals={0: -1.0},
        tracks=np.zeros((3, 2)),
        uncertainties=np.zeros(3),
        visibles=np.array([True, False, True]),
        track_3d=points,
    )
    trace.frontend(frontend, [obj])
    keyframe = SimpleNamespace(obj_id=0, kf_idx=1, pose=obj.pose)
    if with_graph:
        assert graph.update([keyframe]) is graph.result
    points[:] = 5  # Later native map updates cannot rewrite prior diagnostics.
    obj.lost = True
    obj.key_points = np.array([[0.0, 0, 1], [1, 0, 1], [0, 1, 1]])
    obj.valid = np.ones(3, bool)
    obj.track_idx_2_obj_idx = np.arange(3)
    table = SimpleNamespace(obj2track_map=[np.arange(3)], track_2d=np.zeros((3, 2)))
    masks = np.array([[[True, False], [False, True]]])
    manager.sampler = SimpleNamespace(
        reference_depth_edge_filter_event=dict(native_index=0, keep=np.array([True, False]))
    )
    trace.finish([obj], masks=masks, track_table=table)
    obj.key_points[:] = 9
    row = json.loads(path.read_text())
    assert row["sam2_prompts"][0]["points"] == [[1, 2], [3, 4]]
    check = row["promotion_checks"][0]
    assert check["promoted"] and not check["geometric_check_passed"]
    assert check["metadata"]["good"] == 3
    assert check["point_before"][0] == 0 and check["point_after"][0] == 0.01
    assert (row["frame"], row["capture_s"], row["native_index"]) == (248, 4.0, 0)
    assert row["reference_depth_edge_filter"] == dict(native_index=0, keep=[True, False])
    saved = row["objects"][0]
    assert saved["registration_input"]["src_pcd"][0] == [0.0, 0.0, 1.0]
    assert saved["frontend_source_points"][0] == [0.0, 0.0, 1.0]
    assert saved["all_track_indices"] == [0, 1, 2]
    assert saved["all_visibles"] == [True, False, True]
    assert saved["all_current_points"][0] == [0.0, 0.0, 1.0]
    assert saved["map_points"][0] == [0.0, 0.0, 1.0]
    np.testing.assert_array_equal(np.load(tmp_path / row["mask_file"])["masks"], masks)
    assert saved["frontend_pose"][1][3] == 0
    assert saved["published_pose"][1][3] == (0.03 if with_graph else 0)
    assert not saved["frontend_lost"] and saved["lost"]
    assert saved["sdf_refinement"]["before"][0][3] == 0
    assert saved["sdf_refinement"]["after"][0][3] == 0.02
    assert saved["sdf_refinement"]["info"]["seed_cost"] is None
    if with_graph:
        assert row["graph_updates"][0]["inputs"][0]["pose"][1][3] == 0
    else:
        assert row["graph_updates"] == []
