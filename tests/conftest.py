"""Small numerical recordings and shared GPU model setup."""

from __future__ import annotations

import math
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


def make_episode(seed=0, n_steps=20, yaw=0.3, origin=(0.0, 0.0, 0.0)):
    """Explicit numerical recording; no simulator or robot asset required."""
    from alexdoor_xas.action.frames import world_delta_to_frame
    from alexdoor_xas.recording import EpisodeBuffer, EpisodeMeta, EpisodeOutcome, EpisodeStep

    frame = ObjectFrame(np.asarray(origin), rot_z(yaw))
    episode = EpisodeBuffer(
        EpisodeMeta.create(
            task="door_push",
            action_space="A2_ee_delta",
            robot="test_robot",
            scene="synthetic_fixture",
            policy="explicit",
            seed=seed,
            sim_dt=1 / 120,
            control_dt=1 / 60,
            robot_asset_id=TEST_ROBOT_REF.asset_id,
            robot_asset_sha256=TEST_ROBOT_REF.sha256,
        )
    )
    for i in range(n_steps):
        target = np.full(7, 0.001 * (i + seed))
        episode.add_step(
            EpisodeStep(
                t=i / 60,
                action=np.array([0.001 * (seed + 1), 0, 0, 0, 0, 0]),
                proprio={
                    "ee_pos_w": np.array([0.5, seed * 0.01, 1.0]),
                    "ee_quat_w_xyzw": np.array([0.0, 0.0, 0.0, 1.0]),
                    "joint_pos": target,
                    "joint_vel": np.zeros(7),
                    "joint_pos_target": target,
                },
                object_state={
                    "door_angle_rad": i * 0.01,
                    "door_angular_velocity_rad_s": 0.6,
                    "door_rel_pos_x": origin[0],
                    "door_rel_pos_y": origin[1],
                    "door_rel_pos_z": origin[2],
                    "door_yaw_rad": yaw,
                },
                contact={
                    "source": "force_sensor+geometric",
                    "force_n": 2.0,
                    "inferred": True,
                    "sensed": True,
                },
                safety={"controller_phase": "push"},
            )
        )
    episode.extras = {
        "joint_names": [f"joint_{i}" for i in range(7)],
        "arm_joint_ids": list(range(7)),
        "final_joint_pos_target": np.full(7, 0.001 * (n_steps + seed)),
        "action_door_frame": np.stack(
            [world_delta_to_frame(step.action, frame) for step in episode.steps]
        ),
        "door_frame_pos_w": np.asarray(origin),
        "door_frame_quat_w_xyzw": np.array([0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2)]),
        "a4_chunks": [
            {
                "phase": "push",
                "duration_ticks": n_steps,
                "contact_target_panel": [0.036, 0.4, 0.0],
                "motion_hinge_delta_rad": 0.12,
            }
        ],
    }
    episode.set_outcome(EpisodeOutcome(True, 0.12, n_steps, "controller_done", False, False))
    return episode
