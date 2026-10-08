"""Small observed B1 episodes for numerical and GPU regressions."""

import numpy as np

from alexdoor_xas.action.frames import ObjectFrame, rot_z
from alexdoor_xas.assets.identity import RobotAssetRef
from alexdoor_xas.dataset.normalize import DatasetNormStats, NormStats

TEST_ROBOT_REF = RobotAssetRef("test_robot", "a" * 64)


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


def diffusion_action_stats() -> NormStats:
    # Exercise both varying and constant action dimensions.
    low = np.array([-0.015, -0.01, 0.0, 0.0, 0.0, 0.0])
    high = np.array([0.005, 0.013, 0.015, 0.0, 0.0, 0.0])
    return NormStats(
        mean=(low + high) / 2.0,
        std=np.maximum((high - low) / 4.0, 1e-8),
        min=low,
        max=high,
        count=100,
    )


def diffusion_stats() -> DatasetNormStats:
    obs = NormStats(
        mean=np.zeros(14),
        std=np.full(14, 0.5),
        min=np.full(14, -1.0),
        max=np.full(14, 1.0),
        count=100,
    )
    return DatasetNormStats(
        action=diffusion_action_stats(),
        obs=obs,
        obs_keys=("joint_pos", "joint_vel"),
        train_episode_ids=("ep-a",),
        action_space="A2_ee_delta",
    )
