"""Compute experiments must retain the benchmark's geometric/admission controls."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from alexdoor_xas.perception.point2pose_performance import configure_performance


class Params(dict):
    def __getattr__(self, key):
        return self[key]

    def __setattr__(self, key, value):
        self[key] = value


def config():
    return SimpleNamespace(
        tracker=SimpleNamespace(type="tapir", params=Params(resize_height=480, resize_width=480)),
        segmenter=SimpleNamespace(params=Params()),
        register=SimpleNamespace(
            type="svd_cluster_ransac",
            params=Params(
                min_inliers=5,
                inlier_thres=0.004,
                uncer_thres=0.4,
                residual_thres=0.01,
                enable_sdf_refine=True,
            ),
        ),
    )


def test_simplified_svd_requires_effective_fixed_4mm_and_five_inliers():
    c = config()
    configure_performance(c, Path("/models"), {"simplified_svd": True}, diagnostic_only=True)
    assert c.register.params.threshold_method == "fixed"
    assert c.register.params.min_inliers == 5 and c.register.params.inlier_thres == 0.004
    assert c.register.params.uncer_thres == 0.4 and c.register.params.residual_thres == 0.01
    assert not c.register.params.enable_sdf_refine


@pytest.mark.parametrize(
    "controls",
    [
        {"query_chunk_size": 256},
        {"sam2_small": True},
        {"tapir_crop": True, "resolution": 256, "num_pips_iter": 2},
    ],
)
def test_compute_changes_preserve_registration_and_require_diagnostics(controls):
    c = config()
    old = c.register.params.copy()
    with pytest.raises(ValueError, match="diagnostic"):
        configure_performance(c, Path("/models"), controls, diagnostic_only=False)
    configure_performance(c, Path("/models"), controls, diagnostic_only=True)
    assert c.register.params == old
