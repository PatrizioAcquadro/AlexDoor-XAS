"""Operational Purdue configuration; common door placement belongs to Phase 4.1."""

import isaaclab.sim as sim_utils
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils.configclass import configclass


@configclass
class DoorPushPurdueEnvCfg(DirectRLEnvCfg):
    decimation = 2
    episode_length_s = 3600.0
    action_space = 6
    observation_space = 18
    state_space = 0
    sim = sim_utils.SimulationCfg(
        device="cuda:0",
        dt=1 / 120,
        render_interval=2,
    )
    scene = InteractiveSceneCfg(num_envs=1, env_spacing=3.0, replicate_physics=False)
    num_rerenders_on_reset = 2
    reset_camera_frames: int = 8
    action_mode: str = "A2"
    cameras: bool = True
    max_joint_speed: float = 0.5
    max_pos_delta_m: float = 0.01
    max_rot_delta_rad: float = 0.05
    # Safe commissioning pose; Phase 4.1 selects the benchmark ready/parked setup.
    initial_joints: dict = {
        "LEFT_SHOULDER_X": 0.35,
        "LEFT_ELBOW_Y": -1.0,
        "RIGHT_SHOULDER_X": -0.35,
        "RIGHT_ELBOW_Y": -1.0,
    }
