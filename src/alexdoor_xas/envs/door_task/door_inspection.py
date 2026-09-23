"""Door-only legacy preparation measurement, independent of robot execution."""

import torch
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import Articulation, ArticulationCfg, AssetBaseCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg, UsdFileCfg
from isaaclab.utils.configclass import configclass


@configclass
class DoorInspectionCfg(DirectRLEnvCfg):
    decimation = 2
    episode_length_s = 3600.0
    action_space = 1
    observation_space = 2
    state_space = 0
    sense_contacts = False
    sim = SimulationCfg(device="cuda:0", dt=1 / 120, render_interval=2)
    scene = InteractiveSceneCfg(num_envs=1, env_spacing=3.0)
    door_scene = AssetBaseCfg(
        prim_path="/World/envs/env_0/DoorScene", spawn=UsdFileCfg(usd_path="")
    )
    door = ArticulationCfg(
        prim_path="/World/envs/env_0/DoorScene/Door",
        spawn=None,
        actuators={
            "hinge": ImplicitActuatorCfg(joint_names_expr=["Hinge"], stiffness=0.0, damping=4.0)
        },
    )


def tensor(value):
    return value.torch if hasattr(value, "torch") else value


class DoorInspectionEnv(DirectRLEnv):
    def __init__(self, cfg):
        super().__init__(cfg)
        ids, _ = self._door.find_joints("Hinge")
        if len(ids) != 1:
            raise RuntimeError("Door inspection requires one hinge")
        self._hinge_joint_id = ids[0]

    def _setup_scene(self):
        cfg = self.cfg.door_scene
        cfg.spawn.func(cfg.prim_path, cfg.spawn)
        if self.cfg.sense_contacts:
            from isaaclab.sim import schemas
            from pxr import Usd, UsdPhysics

            for prim in Usd.PrimRange(self.sim.stage.GetPrimAtPath(cfg.prim_path)):
                if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                    schemas.activate_contact_sensors(str(prim.GetPath()))
        self._door = Articulation(self.cfg.door)
        self.scene.articulations["door"] = self._door
        self.scene.clone_environments(copy_from_source=False)

    def hinge_state(self):
        return (
            tensor(self._door.data.joint_pos)[:, self._hinge_joint_id],
            tensor(self._door.data.joint_vel)[:, self._hinge_joint_id],
        )

    def _pre_physics_step(self, actions):
        pass

    def _apply_action(self):
        pass

    def _get_observations(self):
        return {"policy": torch.stack(self.hinge_state(), dim=-1)}

    def _get_rewards(self):
        return torch.zeros(1, device=self.device)

    def _get_dones(self):
        zero = torch.zeros(1, dtype=torch.bool, device=self.device)
        return zero, zero.clone()

    def _reset_idx(self, env_ids):
        super()._reset_idx(env_ids)
        q = tensor(self._door.data.default_joint_pos).clone()
        self._door.write_joint_position_to_sim_index(position=q)
        self._door.write_joint_velocity_to_sim_index(velocity=torch.zeros_like(q))
        self._door.set_joint_effort_target_index(target=torch.zeros_like(q))
