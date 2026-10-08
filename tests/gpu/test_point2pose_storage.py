"""Runtime-specific checks separated from portable behavioral tests."""

from collections import namedtuple
from types import SimpleNamespace as NS

import numpy as np
import pytest

from alexdoor_xas.perception.point2pose.renewal import (
    compact_tracker,
)


def test_cuda_compaction_releases_storage_and_preserves_retained_state():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA storage regression requires GPU")
    features = namedtuple("Features", "lowres hires resolutions")
    tracker = NS(
        query_points=torch.arange(90, device="cuda", dtype=torch.float32).reshape(30, 3),
        query_features=features(
            [torch.ones((1, 30, 256), device="cuda")],
            [torch.ones((1, 30, 64), device="cuda")],
            [(256, 256)],
        ),
        _causal_state=[
            {
                "state": torch.arange(30 * 65536, device="cuda", dtype=torch.float32).reshape(
                    1, 30, 65536
                )
            }
        ],
    )
    torch.cuda.synchronize()
    before = torch.cuda.memory_allocated()
    compact_tracker(tracker, np.array([0, 5, 29]))
    torch.cuda.synchronize()
    after = torch.cuda.memory_allocated()
    assert before - after > 6_000_000
    assert tracker._causal_state[0]["state"].shape[1] == 3
    assert tracker._causal_state[0]["state"][0, :, 0].tolist() == [0, 5 * 65536, 29 * 65536]
    assert tracker.query_points[:, 0].tolist() == [0, 15, 87]
    assert tracker.query_features.lowres[0].untyped_storage().nbytes() == 3 * 256 * 4
