"""Real official CUDA fusion/rebuild in its isolated ABI; no simulator is launched."""

import os
import subprocess
from pathlib import Path


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
            f"import runpy; runpy.run_path({str(Path(__file__))!r})['expansion']()",
        ],
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
