"""Diagnostic observation must preserve native decisions and intermediate values."""

import json
from types import SimpleNamespace

import numpy as np

from alexdoor_xas.perception.point2pose_trace import RegistrationTrace


def test_trace_preserves_native_returns_and_copies_before_graph_changes(tmp_path):
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
    pipeline = SimpleNamespace(frontend=SimpleNamespace(register=register), kf_graph=graph)
    path = tmp_path / "registration.jsonl"
    trace = RegistrationTrace(pipeline, path)
    obj = SimpleNamespace(id=0, pose=np.eye(4), lost=False)
    trace.begin(248, 4.0, 0, [obj])
    points = np.array([[0.0, 0, 1], [1, 0, 1], [0, 1, 1]])
    result = register.register(points, points, init_pose=obj.pose, prev_T=obj.pose)
    assert result is register.result and register.refine_return is register.refined
    frontend = SimpleNamespace(
        obj_poses={0: obj.pose},
        valid_key_points={0: points},
        valid_curr_3d={0: points},
        valid_indices={0: np.arange(3)},
        reg_stats={0: result[1]},
        valid_stats={0: dict(confirmed=3)},
        mean_residuals={0: -1.0},
        tracks=np.zeros((3, 2)),
    )
    trace.frontend(frontend, [obj])
    keyframe = SimpleNamespace(obj_id=0, kf_idx=1, pose=obj.pose)
    assert graph.update([keyframe]) is graph.result
    points[:] = 5  # Later native map updates cannot rewrite prior diagnostics.
    obj.lost = True
    trace.finish([obj])
    row = json.loads(path.read_text())
    assert (row["frame"], row["capture_s"], row["native_index"]) == (248, 4.0, 0)
    saved = row["objects"][0]
    assert saved["registration_input"]["src_pcd"][0] == [0.0, 0.0, 1.0]
    assert saved["frontend_source_points"][0] == [0.0, 0.0, 1.0]
    assert saved["frontend_pose"][1][3] == 0 and saved["published_pose"][1][3] == 0.03
    assert not saved["frontend_lost"] and saved["lost"]
    assert saved["sdf_refinement"]["before"][0][3] == 0
    assert saved["sdf_refinement"]["after"][0][3] == 0.02
    assert saved["sdf_refinement"]["info"]["seed_cost"] is None
    assert row["graph_updates"][0]["inputs"][0]["pose"][1][3] == 0
