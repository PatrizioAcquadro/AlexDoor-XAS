"""Small numerical recordings and shared GPU model setup."""

from __future__ import annotations

import sys
from pathlib import Path

WORKTREE_SRC = Path(__file__).resolve().parents[1] / "src"
if str(WORKTREE_SRC) not in sys.path:
    sys.path.insert(0, str(WORKTREE_SRC))

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from alexdoor_xas.action.frames import ObjectFrame, rot_z  # noqa: E402
from alexdoor_xas.assets.identity import RobotAssetRef  # noqa: E402

TEST_ROBOT_REF = RobotAssetRef("test_robot", "a" * 64)


@pytest.fixture
def gpu_models():
    import torch

    if not torch.cuda.is_available():
        pytest.skip("model checks require CUDA")
    previous = torch.get_default_device()
    torch.set_default_device("cuda")
    try:
        yield
    finally:
        torch.set_default_device(previous)


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


def make_b1_episode(binding, *, episode_id="train-1", asset_id="left", sign=1):
    from scipy.spatial.transform import Rotation

    from alexdoor_xas.action.b1 import STAGES, panel_pose
    from alexdoor_xas.dataset.b1 import compile_episode
    from alexdoor_xas.perception.contracts import DoorEstimate
    from alexdoor_xas.policies.observations import PolicyObservation

    frames = [ObjectFrame(np.array([0.1, -0.2, 0.5]), rot_z(0.3)) for _ in range(11)]
    angles = sign * np.arange(11) * 0.002
    tools, observations = [], []
    for i, (frame, angle) in enumerate(zip(frames, angles, strict=True)):
        panel = panel_pose(frame, angle)
        tools.append(
            ObjectFrame(
                panel.point_to_world(np.array([0.04, 0.3, 0.5])),
                panel.rot @ Rotation.from_rotvec([0.1, 0.2, 0.3]).as_matrix(),
            )
        )
        estimate = DoorEstimate(
            i / 60,
            True,
            "observed",
            0.9,
            frame,
            panel.rot,
            float(angle),
            np.array([0.9, 2.0, 0.04]),
            tools[-1].origin,
            tools[-1].rot,
        )
        features = np.r_[np.full(4, 0.01 * i), np.full(9, 0.02 * i), np.zeros(9)]
        observations.append(PolicyObservation(i / 60, i, features, estimate, "observed"))
    return compile_episode(
        episode_id=episode_id,
        asset_id=asset_id,
        asset_splits={"left": "train", "right": "development"},
        observations=observations,
        tools=tools,
        joint_targets=np.full((10, 7), 0.03),
        goals=tools[1:],
        stages=[s for s in STAGES for _ in range(2)],
        binding=binding,
    )
