"""Pure-Python full-contract door environment for software tests."""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass, field
from pathlib import Path

WORKTREE_SRC = Path(__file__).resolve().parents[1] / "src"
if str(WORKTREE_SRC) not in sys.path:
    sys.path.insert(0, str(WORKTREE_SRC))

import numpy as np  # noqa: E402

from alexdoor_xas.action.frames import ObjectFrame, rot_z  # noqa: E402
from alexdoor_xas.adapters.limits import RobotLimitsCfg  # noqa: E402
from alexdoor_xas.assets.identity import RobotAssetRef  # noqa: E402
from alexdoor_xas.kinematics.settle import validate_start_pose_settle  # noqa: E402
from alexdoor_xas.policies.scripted.door_push import DoorPushControllerCfg  # noqa: E402

TEST_ROBOT_LIMITS = RobotLimitsCfg()
TEST_ROBOT_REF = RobotAssetRef("test_robot", "a" * 64)


@dataclass
class SyntheticDoorWorld:
    """Minimal door + EE kinematics: no physics, just enough for the FSM."""

    door_frame: ObjectFrame
    cfg: DoorPushControllerCfg
    gain: float = 0.8
    angle: float = 0.0
    velocity: float = 0.0
    ee_pos_w: np.ndarray = field(default_factory=lambda: np.zeros(3))

    def apply_world(self, delta_world_pos: np.ndarray) -> None:
        """Advance one tick: move the EE in world, resolve panel penetration."""
        self.ee_pos_w = self.ee_pos_w + np.asarray(delta_world_pos, dtype=np.float64)

        ee_door = self.door_frame.point_from_world(self.ee_pos_w)
        ee_panel = rot_z(self.cfg.panel_rotation_sign * self.angle).T @ ee_door
        penetration = self.cfg.surface_x_m(0.0) - float(ee_panel[0])
        in_panel = (
            min(0.0, self.cfg.panel_rotation_sign * self.cfg.panel_width_m)
            <= ee_panel[1]
            <= max(0.0, self.cfg.panel_rotation_sign * self.cfg.panel_width_m)
        )
        if penetration > 0.0 and in_panel:
            previous = self.angle
            self.angle = min(self.angle + self.gain * penetration, math.pi / 2.0)
            self.velocity = self.angle - previous
            # The rigid panel pushes the EE back out to its face.
            ee_panel[0] = self.cfg.surface_x_m(0.0)
            self.ee_pos_w = self.door_frame.point_to_world(
                rot_z(self.cfg.panel_rotation_sign * self.angle) @ ee_panel
            )
        else:
            self.velocity = 0.0


class FakeDoorPushEnv:
    """Complete synthetic stand-in for the adapter environment contract."""

    FORCE_GAIN_N_PER_M = 500.0
    CONTACT_THRESHOLD_N = 1.0
    N_JOINTS = 29
    TARGET_STEP_RAD = 0.001

    class _Cfg:
        class _Sim:
            dt = 1 / 120

        sim = _Sim()
        decimation = 2
        max_pos_delta_m = 0.02
        max_rot_delta_rad = 0.05
        start_pose_tolerance_m = 0.01

    cfg = _Cfg()

    def __init__(
        self,
        yaw_rad: float = 0.0,
        origin: tuple[float, float, float] = (0.0, 0.0, 0.0),
        controller_cfg: DoorPushControllerCfg | None = None,
        start_door_frame: tuple[float, float, float] = (0.7, 0.2, 0.0),
    ):
        self._yaw = yaw_rad
        self._frame = ObjectFrame(origin=np.asarray(origin, dtype=np.float64), rot=rot_z(yaw_rad))
        self._controller_cfg = controller_cfg or DoorPushControllerCfg()
        self._start_door_frame = np.asarray(start_door_frame, dtype=np.float64)
        self.world: SyntheticDoorWorld | None = None
        self._ticks = 0
        self._contact_state_calls = 0
        self._last_settle_report: dict | None = None

    def reset(self, seed: int | None = None):
        self._ticks = 0
        self._contact_state_calls = 0
        self._last_settle_report = None
        self.world = SyntheticDoorWorld(door_frame=self._frame, cfg=self._controller_cfg)
        self.world.ee_pos_w = self._frame.point_to_world(self._start_door_frame)
        return {"policy": None}, {}

    def step(self, action):
        import torch

        assert self.world is not None, "reset() must be called before step()"
        self._ticks += 1
        delta = action.detach().cpu().numpy().reshape(-1)
        clamped = delta.copy()
        clamped[:3] = np.clip(clamped[:3], -self.cfg.max_pos_delta_m, self.cfg.max_pos_delta_m)
        self.world.apply_world(clamped[:3])
        obs = {"policy": None}
        zero = torch.zeros(1)
        return obs, zero, torch.zeros(1, dtype=torch.bool), torch.zeros(1, dtype=torch.bool), {}

    def door_frame_pose_w(self):
        import torch

        half = self._yaw / 2.0
        quat_xyzw = torch.tensor([[0.0, 0.0, math.sin(half), math.cos(half)]])
        return torch.tensor(self._frame.origin, dtype=torch.float64).reshape(1, 3), quat_xyzw

    def hinge_state(self):
        import torch

        assert self.world is not None
        return (
            torch.tensor([self.world.angle], dtype=torch.float64),
            torch.tensor([self.world.velocity], dtype=torch.float64),
        )

    def ee_pose_w(self):
        import torch

        assert self.world is not None
        return (
            torch.tensor(self.world.ee_pos_w, dtype=torch.float64).reshape(1, 3),
            torch.tensor([[0.0, 0.0, 0.0, 1.0]], dtype=torch.float64),
        )

    def set_ee_pose_w(self, pos_w, quat_w) -> None:
        assert self.world is not None
        requested = pos_w.detach().cpu().numpy().reshape(3).astype(np.float64)
        self.world.ee_pos_w = requested.copy()
        self._last_settle_report = validate_start_pose_settle(
            requested,
            self.world.ee_pos_w,
            settle_ticks_used=0,
            max_settle_ticks=90,
            tolerance_m=self.cfg.start_pose_tolerance_m,
        ).to_dict()

    def start_pose_settle_report(self):
        return self._last_settle_report

    def contact_state(self):
        import torch

        self._contact_state_calls += 1
        force_n = self._contact_force_n()
        return (
            torch.tensor([[force_n, 0.0, 0.0]], dtype=torch.float64),
            torch.tensor([force_n >= self.CONTACT_THRESHOLD_N]),
        )

    def robot_joint_state(self):
        targets = np.zeros(self.N_JOINTS)
        targets[self.arm_joint_ids()] = self.TARGET_STEP_RAD * self._ticks
        return {
            "joint_pos": targets.copy(),
            "joint_vel": np.zeros(self.N_JOINTS),
            "joint_pos_target": targets,
        }

    def robot_joint_names(self):
        return [f"JOINT_{i}" for i in range(self.N_JOINTS)]

    def arm_joint_ids(self):
        return list(range(6))

    def robot_joint_limits(self):
        return {
            "joint_pos_limits": np.stack(
                [np.full(self.N_JOINTS, -2.5), np.full(self.N_JOINTS, 2.5)], axis=1
            ),
            "joint_vel_limits": np.full(self.N_JOINTS, 10.0),
        }

    def robot_base_pos_w(self):
        import torch

        return torch.zeros((1, 3), dtype=torch.float64)

    @property
    def episode_length_buf(self):
        import torch

        return torch.tensor([self._ticks])

    def robot_asset_provenance(self):
        return TEST_ROBOT_REF.to_dict()

    def ik_clamp_telemetry(self):
        return {
            "joints": {},
            "n_solve_ticks": self._ticks,
            "max_excess_rad": 0.0,
            "clamp_ticks_total": 0,
        }

    def _contact_force_n(self) -> float:
        assert self.world is not None
        cfg = self.world.cfg
        ee_door = self.world.door_frame.point_from_world(self.world.ee_pos_w)
        ee_panel = rot_z(self.world.angle).T @ ee_door
        depth = cfg.surface_x_m(cfg.contact_eps_m) - float(ee_panel[0])
        within = 0.0 <= ee_panel[1] <= cfg.panel_width_m
        if depth <= 0.0 or not within:
            return 0.0
        return self.FORCE_GAIN_N_PER_M * depth


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
