"""Alex003 assets and solved-contact access for the upstream Newton backend."""

import xml.etree.ElementTree as ET
from pathlib import Path
from types import SimpleNamespace

import warp as wp
from ihmc_alex_isaaclab.robots.purdue_frames import ALEX_PURDUE_ASSET_ROOT
from isaaclab.actuators import IdealPDActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.sim import UsdFileCfg

from .steel_contacts import SteelPurdueViews

ROBOT_ASSET = "alex_purdue_wsg32_umi_v1_full_convex"
PEDESTAL_ASSET = "purdue_alex003_pedestal"


def converted_usd(cache, kind, name):
    return UsdFileCfg(
        usd_path=str((Path(cache) / kind / name / f"{name}.usda").resolve(strict=True))
    )


def robot_config(cache):
    """Use the converted Alex asset, its URDF limits and the Alex Purdue PD gains."""
    urdf = ALEX_PURDUE_ASSET_ROOT / "urdf/baseline" / f"{ROBOT_ASSET}.urdf"
    limits = {joint.get("name"): joint.find("limit") for joint in ET.parse(urdf).findall("joint")}
    groups = {
        "neck": (("NECK_Z", "NECK_Y"), (5.0, 5.0), (1.0, 1.0), 0.01),
        "arms": (
            (
                "SHOULDER_Y",
                "SHOULDER_X",
                "SHOULDER_Z",
                "ELBOW_Y",
                "WRIST_Z",
                "WRIST_X",
                "GRIPPER_Y",
            ),
            (26.78, 26.78, 23.76, 23.76, 5.0, 5.0, 2.0),
            (8.0, 8.0, 4.0, 4.0, 1.0, 1.0, 0.5),
            0.01,
        ),
        "wsg32": (("WSG32_JAW_OPENING",), (2000.0,), (60.0,), 0.001),
    }
    actuators = {}
    for group, (suffixes, stiffness, damping, armature) in groups.items():
        gains = {
            name: (kp, kd)
            for suffix, kp, kd in zip(suffixes, stiffness, damping, strict=True)
            for name, limit in limits.items()
            if limit is not None and (name == suffix or name.endswith("_" + suffix))
        }
        velocity = {name: float(limits[name].get("velocity")) for name in gains}
        effort = {name: float(limits[name].get("effort")) for name in gains}
        actuators[group] = IdealPDActuatorCfg(
            joint_names_expr=list(gains),
            stiffness={name: pair[0] for name, pair in gains.items()},
            damping={name: pair[1] for name, pair in gains.items()},
            actuator_velocity_limit=velocity,
            actuator_effort_limit=effort,
            joint_velocity_limit=velocity,
            joint_effort_limit=effort,
            armature=armature,
        )
    return ArticulationCfg(
        spawn=converted_usd(cache, "robot", ROBOT_ASSET),
        soft_joint_pos_limit_factor=1.0,
        actuators=actuators,
    ), urdf


class NativeSteelPurdueViews(SteelPurdueViews):
    """Present the official backend's solved rows to the shared contact observer."""

    def __init__(self, sim, panel_path):
        manager = sim.backend_manager
        backend = manager.backend
        manager.add_contact_sensor(body_names_expr=panel_path)
        poses = wp.clone(backend.state_0.body_q)
        # Contact points are local to the bodies at collision time.
        manager.register_pre_actuator_callback(lambda: wp.copy(poses, backend.state_0.body_q))
        self.backend = backend
        super().__init__(
            SimpleNamespace(
                _model=backend.model,
                _state_0=backend.state_0,
                _contacts=backend.contacts,
                _solver=backend.solver,
                _solver_dt=backend.solver_dt,
                _sim=sim,
                observables=backend.contact_observables,
                contact_poses=poses,
            )
        )

    def reset(self):
        # The native backend resets solver state on the subsequent forward().
        self.backend.solver.check_constraint_capacity()
        self.backend.contacts.clear()
        self.snapshot.invalidate()
