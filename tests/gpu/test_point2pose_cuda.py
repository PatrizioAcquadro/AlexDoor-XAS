"""Real official CUDA fusion/rebuild in its isolated ABI; no simulator is launched."""

import os
import subprocess
from pathlib import Path


def readonly_launch():
    """Compare every fused value and caller buffer against the pre-patch CUDA code."""
    import ast

    import numpy as np
    import point2pose.modules.reconstruction.sdf_builder as native
    import pycuda.driver as cuda

    source = Path(native.__file__).with_suffix(".py.before-tracking-fixes")
    node = next(
        n
        for n in ast.parse(source.read_text()).body
        if isinstance(n, ast.ClassDef) and n.name == "TSDFVolume"
    )
    namespace = vars(native).copy()
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), "exec"), namespace)
    bounds = np.array([[-0.1, 0.1], [-0.12, 0.12], [0.9, 1.1]])
    volumes = [cls(bounds.copy(), 0.02) for cls in (namespace["TSDFVolume"], native.TSDFVolume)]
    rgb = np.arange(30 * 40 * 3, dtype=np.uint8).reshape(30, 40, 3)
    depth = np.linspace(0.95, 1.05, 30 * 40, dtype=np.float32).reshape(30, 40)
    intrinsics = np.array([[50, 0, 20], [0, 50, 15], [0, 0, 1]], np.float32)
    pose = np.eye(4, dtype=np.float32)
    before = [a.copy() for a in (rgb, depth, intrinsics, pose)]
    readbacks = []
    copy = cuda.memcpy_dtoh

    def observe(destination, source):
        readbacks.append(destination.nbytes)
        return copy(destination, source)

    cuda.memcpy_dtoh = observe
    try:
        for multiloop in (False, True):
            counts = []
            for volume in volumes:
                if multiloop:
                    volume._max_gpu_grid_dim = np.ones(3, dtype=int)
                    volume._max_gpu_threads_per_block = 128
                    volume._n_gpu_loops = int(np.ceil(np.prod(volume._vol_dim) / 128))
                readbacks.clear()
                for weight in (1.0, 2.0):
                    volume.integrate(rgb, depth, intrinsics, pose, obs_weight=weight)
                counts.append((len(readbacks), sum(readbacks)))
            loops = volumes[0]._n_gpu_loops
            assert counts[0][0] - counts[1][0] == 2 * loops * 6
            assert counts[0][1] - counts[1][1] == 2 * loops * (12 + 36 + 64 + 24 + 2 * depth.nbytes)
            for old, new in zip(volumes[0].get_volume(), volumes[1].get_volume(), strict=True):
                np.testing.assert_array_equal(old, new, strict=True)
            np.testing.assert_array_equal(
                volumes[0].get_weight_volume(), volumes[1].get_weight_volume(), strict=True
            )
        for old, new in zip(before, (rgb, depth, intrinsics, pose), strict=True):
            np.testing.assert_array_equal(old, new, strict=True)

        # Both paths must raise within integrate, before its caller can publish success.
        def fail(*args, **kwargs):
            raise RuntimeError("launch_failed")

        for volume in volumes:
            volume._cuda_integrate = fail
            try:
                volume.integrate(rgb, depth, intrinsics, pose)
            except RuntimeError as error:
                assert str(error) == "launch_failed"
            else:
                raise AssertionError("CUDA failure was swallowed")
    finally:
        cuda.memcpy_dtoh = copy


def expansion():
    from types import SimpleNamespace

    import numpy as np
    import torch
    from point2pose.data_types.frame import Frame
    from point2pose.modules.object.object import Object
    from point2pose.modules.reconstruction.sdf_builder import SDFBuilder

    from alexdoor_xas.perception.point2pose.scale import panel_sdf_builder

    assert torch.cuda.is_available()
    builder = panel_sdf_builder(SDFBuilder, 0.004)(
        dict(
            sdf_backend="legacy",
            sdf_use_gpu=True,
            sdf_voxel_size=0.005,
            sdf_min_points=300,
            sdf_bounds_padding=0.05,
            sdf_filter_keep_percentile=98,
        )
    )
    obj = Object(0)
    obj.add_key_points(
        np.array([[0, 0, 1.0], [0.2, 0.2, 1]]),
        np.zeros(2),
        np.ones(2, bool),
        np.arange(2),
        frame_id=0,
    )
    sparse = obj.key_points
    mapping = obj.track_idx_2_obj_idx
    graph = obj.keyframes
    k = np.array([[50, 0, 50], [0, 50, 50], [0, 0, 1.0]])
    depth = np.ones((100, 100), dtype=np.float32)
    for index, width in enumerate((30, 80)):
        mask = np.zeros((1, 1, 100, 100), dtype=bool)
        mask[0, 0, 10:90, 10:width] = True
        v, u = np.nonzero(mask[0, 0])
        pts = np.c_[(u - 50) / 50, (v - 50) / 50, np.ones(len(u))]
        frame = Frame(
            index,
            np.zeros((100, 100, 3), np.uint8),
            depth,
            torch.as_tensor(mask, device="cuda"),
            k,
            1.0,
        )
        kf = SimpleNamespace(pose=np.eye(4), dense_pts=pts, frame=frame, obj_id=0)
        obj.keyframes.append(kf)
        assert builder.integrate_keyframe(obj, kf)
    assert builder.rebuilds == 1 and obj.sdf_num_integrated == 2
    assert obj.sdf_volume.gpu_mode and obj.sdf_volume.get_weight_volume().max() >= 2
    assert (
        obj.key_points is sparse and obj.track_idx_2_obj_idx is mapping and obj.keyframes is graph
    )
    assert obj.sdf_volume._vol_bnds[0, 1] > 0.55  # New observed extent, not a 25 cm crop.
    # Enforce explicit failure before allocating an impossible observed extent.
    builder.envelopes[obj.id] = np.array([[-1e3, 1e3]] * 3)
    with_error = False
    try:
        builder._init_object_volume(obj, pts)
    except MemoryError as error:
        with_error = "tsdf_memory_budget_exceeded" in str(error)
    assert with_error


def test_official_cuda_expansion_preserves_sparse_state():
    import pytest

    root = Path(__file__).resolve().parents[2]
    runtime = root / "models/perception/point2pose/runtime/bin/python"
    if not runtime.exists():
        pytest.skip("Isolated official Point2Pose runtime is not installed")
    env = dict(
        os.environ,
        PATH=f"{os.environ.get('CUDA_HOME', '/usr/local/cuda')}/bin:{os.environ.get('PATH', '')}",
        PYTHONPATH=f"{root}/src:{root}/models/perception/point2pose/upstream",
        NVCC_PREPEND_FLAGS="-ccbin /usr/bin/g++-11",
        OPENBLAS_NUM_THREADS="1",
    )
    env.pop("LD_LIBRARY_PATH", None)
    env.pop("PYTHONHOME", None)
    result = subprocess.run(
        [
            str(runtime),
            "-c",
            f"import runpy; checks = runpy.run_path({str(Path(__file__))!r}); "
            "checks['readonly_launch'](); checks['expansion']()",
        ],
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
