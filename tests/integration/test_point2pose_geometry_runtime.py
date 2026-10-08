"""Runtime-specific checks separated from portable behavioral tests."""

from pathlib import Path

import numpy as np
import pytest


def test_lifting_optimization_matches_official_measured_and_completed_depth():
    path = Path(__file__).resolve().parents[2] / "models/perception/point2pose/upstream"
    if not path.exists():
        pytest.skip("Official runtime not installed; pure contracts remain covered")
    import sys

    sys.path.insert(0, str(path))
    from point2pose.utils.camera import convert_pixel_to_world

    from alexdoor_xas.perception.point2pose.compat import measured_depth_lifting

    optimized = measured_depth_lifting(convert_pixel_to_world, (0.1, 4.0))
    depth = np.ones((32, 32), dtype=np.float32)
    pixels = np.array([[2, 2], [15, 15], [0, 31], [-1, 10], [31, 0]])
    k = np.array([[200, 0, 16], [0, 200, 16], [0, 0, 1.0]])
    for missing in (False, True):
        depth[15, 15] = np.nan if missing else 1
        for uncertainty in (False, True):
            args = dict(
                pixel=pixels,
                depth_image=depth,
                cam_intrinsics=k,
                min_depth=0.1,
                max_depth=4.0,
                fill_missing_depth=True,
                window_size=5,
                compute_depth_uncertainty=uncertainty,
            )
            actual, expected = optimized(**args), convert_pixel_to_world(**args)
            for a, e in zip(actual, expected, strict=True):
                if a is None:
                    assert e is None
                else:
                    np.testing.assert_allclose(a, e, atol=0, rtol=0, equal_nan=True)
