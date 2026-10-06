"""Exercise the patched native numerical methods without loading vision models."""

import ast
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Optional

import numpy as np
import pytest

from alexdoor_xas.perception.point2pose_patches import NATIVE_FIXES, patch_tracking

REGISTER_PATH = "upstream/point2pose/modules/register/svd_cluster_ransac_register.py"
FRONTEND_PATH = "upstream/point2pose/pipeline/components/front_end.py"
CRITERION_PATH = "upstream/point2pose/modules/criterion/rotation_thres_and_min_num_criterion.py"
PATCH_PATHS = (REGISTER_PATH, FRONTEND_PATH, CRITERION_PATH)


@pytest.fixture
def native_root(tmp_path, monkeypatch):
    installed = Path(__file__).resolve().parents[1] / "models/perception/point2pose"
    if not (installed / REGISTER_PATH).exists():
        pytest.skip("Pinned native source is not installed")
    monkeypatch.syspath_prepend(str(installed / "upstream"))
    for relative in PATCH_PATHS:
        source = installed / relative
        original = source.with_suffix(".py.before-tracking-fixes")
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text((original if original.exists() else source).read_text())
    return tmp_path


def native_class(path, name, namespace):
    # Importing point2pose.modules eagerly imports every GPU/model dependency.
    # Execute the actual class body with its numerical dependencies instead.
    node = next(
        n
        for n in ast.parse(path.read_text()).body
        if isinstance(n, ast.ClassDef) and n.name == name
    )
    node.decorator_list = []
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)
    return namespace[name]


def test_installer_preserves_originals_and_is_idempotent(native_root):
    originals = {r: (native_root / r).read_text() for r in PATCH_PATHS}
    assert patch_tracking(native_root) == NATIVE_FIXES
    modified = {r: (native_root / r).read_text() for r in originals}
    assert all(modified[r] != originals[r] for r in originals)
    assert patch_tracking(native_root) == NATIVE_FIXES
    for relative, original in originals.items():
        path = native_root / relative
        assert path.read_text() == modified[relative]
        assert path.with_suffix(".py.before-tracking-fixes").read_text() == original


def test_unexpected_source_fails_before_any_edit(native_root):
    register = native_root / REGISTER_PATH
    original = register.read_text()
    (native_root / FRONTEND_PATH).write_text("unexpected upstream source")
    with pytest.raises(ValueError, match="Unexpected pinned"):
        patch_tracking(native_root)
    assert register.read_text() == original
    assert not register.with_suffix(".py.before-tracking-fixes").exists()


@pytest.fixture
def register(native_root):
    from point2pose.core.base_register import Register
    from point2pose.utils.transform import transform_pts

    patch_tracking(native_root)
    cls = native_class(
        native_root / REGISTER_PATH,
        "SVDClusterRANSACRegister",
        dict(np=np, Register=Register, transform_pts=transform_pts),
    )
    return cls(
        dict(
            min_inliers=5, inlier_thres=0.004, select_method="inlier_count", enable_sdf_refine=True
        )
    )


@pytest.mark.parametrize(
    "refinement,enabled", [(0.001, True), (0.1, True), (0.0, True), (0.0, False)]
)
def test_returned_sdf_pose_residuals_and_support_gate(register, refinement, enabled):
    source = np.array(
        [[0, 0, 1], [0.1, 0, 1], [0, 0.1, 1], [0.1, 0.1, 1], [0.2, 0, 1], [0, 0.2, 1]]
    )
    cluster_pose, refined_pose = np.eye(4), np.eye(4)
    refined_pose[0, 3] = refinement
    target = source.copy()
    if refinement == 0.001:
        target[:, 0] += refinement
        target[:, 2] += 0.0005
    candidates = iter([dict(T=cluster_pose, ninliers=6, mean_res=0.001), None])
    register._RANSAC = lambda **kwargs: next(candidates)
    register._enable_sdf_refine = enabled
    register._maybe_refine_with_sdf = lambda **kwargs: (refined_pose, dict(applied=True))
    pose, stats = register.register(source, target, init_pose=np.eye(4))
    gate = enabled and refinement == 0.1
    np.testing.assert_array_equal(pose, cluster_pose if gate else refined_pose)
    np.testing.assert_array_equal(cluster_pose, np.eye(4))
    residuals = np.linalg.norm(source @ pose[:3, :3].T + pose[:3, 3] - target, axis=1)
    np.testing.assert_allclose(stats["residuals"], residuals, atol=1e-15)
    np.testing.assert_array_equal(stats["inliers"], residuals <= 0.004)
    np.testing.assert_array_equal(register._last_selected_T, pose)
    assert stats["final_inlier_gate"]["applied"] == gate
    if gate:
        assert stats["final_inlier_gate"]["fallback_to"] == "cluster_selected_T"
        assert stats["final_inlier_gate"]["fallback_ninliers"] == 6


def test_no_cluster_retains_pose_and_failure_evidence(register):
    previous = np.eye(4)
    previous[1, 3] = 0.02
    register._RANSAC = lambda **kwargs: None
    pose, stats = register.register(np.ones((10, 3)), np.ones((10, 3)), init_pose=previous)
    np.testing.assert_array_equal(pose, previous)
    assert stats["best_cluster_idx"] == -1 and stats["clusters"] == []
    assert not stats["inliers"].any()
    np.testing.assert_array_equal(stats["residuals"], np.full(10, -1.0))


@pytest.mark.parametrize("enabled,refit_shift", [(False, 0.005), (True, 0.005), (True, 0.001)])
def test_refit_rollback_revalidates_own_seed_and_preserves_valid_refit(
    register, monkeypatch, enabled, refit_shift
):
    from point2pose.utils.transform import transform_pts

    source = np.array([[i * 0.01, (i % 3) * 0.02, 1] for i in range(7)])
    init = np.eye(4)
    init[1, 3] = 0.03
    p0 = transform_pts(init, source)
    target = p0.copy()
    target[:, 0] += [0, 0, 0, 0, 0.0039, 0, 0.02]
    remaining = np.array([True, True, True, True, True, False, True])
    before = remaining.copy()
    refit = np.eye(4)
    refit[0, 3] = refit_shift
    fits = iter([np.eye(4), refit])
    register._svd_fit = lambda *args: next(fits)
    register._is_degenerate_sample = lambda points: False
    register._ransac_iters = 1
    register._sample_size = 4
    register._refit_seed_rollback = enabled
    monkeypatch.setattr(np.random, "choice", lambda *args, **kwargs: np.arange(4))
    candidate = register._RANSAC(p0, target, None, remaining, init)
    if not enabled:
        assert candidate is None
        np.testing.assert_array_equal(remaining, before)
        return
    rolled_back = refit_shift == 0.005
    pose = init if rolled_back else refit @ init
    np.testing.assert_array_equal(candidate["T"], pose)
    residuals = np.linalg.norm(transform_pts(pose, source) - target, axis=1)
    expected = np.flatnonzero(before & (residuals <= 0.004))
    np.testing.assert_array_equal(candidate["inliers"], expected)
    assert candidate["ninliers"] == len(expected) == 5
    assert candidate["mean_res"] == pytest.approx(residuals[expected].mean())
    assert candidate["refit_seed_rollback"] == rolled_back
    assert candidate["refit_ninliers"] == (1 if rolled_back else 5)
    np.testing.assert_array_equal(remaining, before & ~(residuals <= 0.004))


@pytest.fixture
def frontend(native_root):
    from point2pose.data_types.front_end_result import FrontEndResult
    from point2pose.utils.camera import convert_pixel_to_world

    patch_tracking(native_root)
    cls = native_class(
        native_root / FRONTEND_PATH,
        "FrontEnd",
        dict(
            np=np,
            time=time,
            PointTrackTable=object,
            FrontEndResult=FrontEndResult,
            Tuple=tuple,
            Optional=Optional,
            Dict=dict,
            Any=Any,
            convert_pixel_to_world=convert_pixel_to_world,
        ),
    )
    fe = cls.__new__(cls)
    fe.__dict__.update(
        use_segmenter=False,
        num_obj=1,
        min_depth=0.1,
        max_depth=4.0,
        fill_missing_depth=False,
        fill_missing_depth_window_size=3,
        fill_missing_depth_min_neighbors=1,
        frame_reg_mode="f2m",
        reg_uncer_thres=0.4,
        reg_mask_border_margin_px=0,
        reg_residual_thres=0.01,
        debug_level=0,
        save_cropped_pcd=False,
        prev_frame=None,
        prev_residuals={},
        prev_inlier_counts={},
    )
    uv = np.c_[np.arange(10) + 2, np.full(10, 5)]
    fe.tracker = SimpleNamespace(track_once=lambda frame: (uv, np.zeros(10), np.ones(10)))
    fe._extract_valid_key_points_mask_remove = lambda *args, **kwargs: (
        np.arange(10),
        np.ones((10, 3)),
        np.ones((10, 3)),
        np.ones(10, bool),
        {},
    )
    fe._update_pose_history = lambda *args: None
    return fe


def frontend_step(fe, obj, residual, inliers, rejected=False):
    pose = obj.pose.copy()
    if inliers:
        pose[0, 3] = 0.002
    stats = dict(inliers=np.arange(10) < inliers, residuals=np.full(10, residual))
    fe.register = SimpleNamespace(_min_inliers=5, register=lambda **kwargs: (pose, stats))
    fe._apply_pose_jump_guard = lambda **kwargs: (
        obj.pose if rejected else pose,
        rejected,
        dict(rejected=rejected),
    )
    frame = SimpleNamespace(
        id=1 if fe.prev_frame is None else fe.prev_frame.id + 1,
        depth=np.ones((20, 20)),
        intrinsics=np.eye(3),
        depth_factor=1.0,
        mask=np.ones((1, 1, 20, 20)),
    )
    return fe.step(frame, SimpleNamespace(obj2track_map=[np.arange(10)]), [obj])


def test_native_loss_and_same_object_recovery(frontend):
    obj = SimpleNamespace(id=0, pose=np.eye(4), lost=False)
    first = frontend_step(frontend, obj, -1.0, 0)
    assert obj.lost and first.mean_residuals[0] == -1.0
    np.testing.assert_array_equal(first.obj_poses[0], obj.pose)
    # f2m must still run on this very same lost object, without reselection/reset.
    second = frontend_step(frontend, obj, 0.001, 6)
    assert not obj.lost and obj.id == 0
    np.testing.assert_array_equal(second.obj_poses[0][:3, 3], [0.002, 0, 0])


@pytest.mark.parametrize(
    "residual,inliers,rejected,lost",
    [
        (0.001, 4, False, True),
        (0.001, 5, True, True),
        (0.01, 6, False, True),
        (np.nan, 6, False, True),
        (np.inf, 6, False, True),
        (0.0, 5, False, False),
    ],
)
def test_native_lost_is_recomputed_from_current_supported_result(
    frontend, residual, inliers, rejected, lost
):
    obj = SimpleNamespace(id=0, pose=np.eye(4), lost=not lost)
    frontend_step(frontend, obj, residual, inliers, rejected)
    assert obj.lost is lost


@pytest.mark.parametrize(
    "ninliers,angle,area,expected",
    [(9, 0, 101, True), (10, 0, 101, False), (9, 0, 100, False), (20, 16, 101, True)],
)
def test_inlier_trigger_preserves_mask_and_view_thresholds(
    native_root, ninliers, angle, area, expected
):
    import torch
    from point2pose.core.base_criterion import SampleCriterion

    patch_tracking(native_root)
    cls = native_class(
        native_root / CRITERION_PATH,
        "RotationThresholdAndMinNumCriterion",
        dict(np=np, torch=torch, SampleCriterion=SampleCriterion, CriterionContext=object),
    )
    criterion = cls(dict(max_angle_deg=15, min_num_pts=10, min_mask_area=100))
    theta = np.deg2rad(angle)
    pose = np.eye(4)
    pose[:3, :3] = [
        [np.cos(theta), 0, np.sin(theta)],
        [0, 1, 0],
        [-np.sin(theta), 0, np.cos(theta)],
    ]
    context = SimpleNamespace(
        objects=[SimpleNamespace(pose=pose)],
        frame=SimpleNamespace(id=1, mask=torch.ones((1, 1, 1, area))),
        reg_stats={0: dict(correspond_curr3d=np.zeros((30, 3)), inliers=np.arange(30) < ninliers)},
    )
    criterion.initialize(context)
    assert criterion.check_sample_criterion(context, 0) == expected
