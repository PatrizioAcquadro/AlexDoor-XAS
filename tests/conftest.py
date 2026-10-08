"""Shared B1 contract fixture; no simulator or model initialization."""

from __future__ import annotations

import sys
from pathlib import Path

WORKTREE_SRC = Path(__file__).resolve().parents[1] / "src"
if str(WORKTREE_SRC) not in sys.path:
    sys.path.insert(0, str(WORKTREE_SRC))

import pytest  # noqa: E402


@pytest.fixture
def b1_binding():
    import json

    from alexdoor_xas.policies.common.b1_contract import RELEASE_SCHEMA, PerceptionBinding

    config = dict(
        visual_dims=[2, 2],
        max_gap_s=0.15,
        warmup_s=0.4,
        inspection=json.loads(
            (WORKTREE_SRC.parent / "configs/perception_inspection.json").read_text()
        ),
    )
    # Numerical fixture only: this is not a release of any existing checkpoint.
    return PerceptionBinding.from_dict(
        dict(
            schema=RELEASE_SCHEMA,
            artifacts={"geometry": "a" * 64, "visual": "b" * 64},
            config=config,
            offline_passed=True,
            dynamic_passed=True,
            frozen=True,
        )
    )
