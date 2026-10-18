"""Small observed B1 episodes for numerical and GPU regressions."""

from itertools import product
from types import SimpleNamespace

import numpy as np
from scipy.spatial import ConvexHull

from alexdoor_xas.action.b1 import apply_pose_delta
from alexdoor_xas.action.frames import ObjectFrame, rot_z
from alexdoor_xas.assets.identity import RobotAssetRef
from alexdoor_xas.dataset.b1 import compile_episode
from alexdoor_xas.dataset.normalize import DatasetNormStats, NormStats
from alexdoor_xas.perception.contracts import DoorEstimate
from alexdoor_xas.policies.observations import (
    B1Observer,
    FeatureBlock,
    PolicyObservation,
    VisualEncoding,
)
from alexdoor_xas.qualification.door_geometry import PreparedDoor
from alexdoor_xas.recording.b1 import OBS_KEYS

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


def fake_estimator(binding):
    estimator = SimpleNamespace(binding=binding, reason="observed")

    def reset():
        estimator.encoding = None
        estimator.calls = []

    def update(obs):
        estimator.calls.append(obs)
        estimator.encoding = np.arange(binding.obs_dim - 18)
        return DoorEstimate(
            obs["time_s"],
            estimator.reason == "observed",
            estimator.reason,
            0.8,
            ObjectFrame(np.zeros(3), np.eye(3)),
            np.eye(3),
            0.0,
            np.array([0.9, 2.0, 0.04]),
            np.array([0.0, 0.3, 1.09]),
            np.eye(3),
        )

    estimator.reset, estimator.update = reset, update
    return estimator


def sample(t, frame):
    result = dict.fromkeys(OBS_KEYS, None)
    result.update(time_s=t, frame=frame, joint_position=np.zeros(9), joint_velocity=np.zeros(9))
    return result


def operational_provider(binding, count=1):
    from dataclasses import replace

    from alexdoor_xas.perception.contracts import OPERATIONAL_V1
    from perception_helpers import operational_estimate

    recipe = SimpleNamespace(
        config=binding.config | {"geometry_profile": OPERATIONAL_V1}, obs_dim=binding.obs_dim
    )
    provider = SimpleNamespace(binding=recipe, generation=-1, between=False, old_result=None)

    def reset():
        provider.generation += 1
        provider.encoding = None
        provider.calls = []
        provider.selection_time = None

    def update(sensor):
        provider.calls.append(sensor)
        if provider.between:
            return DoorEstimate(sensor["time_s"], False, "between_inference_ticks")
        estimate = provider.old_result or operational_estimate(
            sensor["time_s"], provider.generation, count
        )
        if provider.selection_time is None:
            provider.selection_time = sensor["time_s"]
        state = estimate.operational
        estimate = replace(
            estimate,
            operational=replace(
                state, contact=replace(state.contact, selected_s=provider.selection_time)
            ),
        )
        provider.encoding = np.arange(recipe.obs_dim - 18)
        provider.last_estimate = estimate
        return estimate

    provider.reset, provider.update = reset, update
    return provider


class NumericalIO:
    robot_asset = TEST_ROBOT_REF
    dt = 0.1
    max_ticks = 20

    def __init__(self, binding):
        self.binding = binding
        self.fail_after = None
        self.reset()

    def reset(self):
        self.clock = 0
        self.commands = []
        self.held = 0
        self.stops = 0
        self.pose = ObjectFrame(np.zeros(3), np.eye(3))
        self.neck = []

    def observe(self):
        return sample(self.clock * self.dt, self.clock)

    def tool_pose(self):
        return self.pose

    def set_neck_target(self, value):
        self.neck.append(value)

    def hold_step(self):
        self.held += 1
        self.clock += 1

    def execute(self, command):
        self.commands.append(command)
        if command.kind == "pose":
            self.pose = command.value
        elif command.kind == "delta":
            self.pose = apply_pose_delta(self.pose, command.value)
        self.clock += 1

    def safety_reason(self, stage):
        return "force_limit" if self.fail_after == len(self.commands) else ""

    def stop(self):
        self.stops += 1


def make_door(tmp_path, hand):
    folder = tmp_path / hand
    (folder / "prepared").mkdir(parents=True)
    (folder / "prepared/door.usda").touch()
    sign = 1 if hand == "left" else -1
    record = dict(
        handedness=hand,
        dimensions_m=dict(width_m=0.9, height_m=2, thickness_m=0.06),
        hinge_m=[0.11, sign * 0.46, 0],
        panel_center_m=[0.07, 0, 1.03],
        mechanical_limit_deg=74.5,
        usd="prepared/door.usda",
    )
    door = PreparedDoor(folder, record, {"components": {"Panel": [0]}})
    points = np.array(list(product([0.04, 0.10], [-0.45, 0.45], [0.03, 2.03])))
    door._leaf_planes = [ConvexHull(points).equations]
    door.shapes = {"Panel": [points], "Frame": [points + [0.1, 0.5, 0]], "Handle": []}
    return door


def operational_observation(estimate, time=0):
    from alexdoor_xas.perception.contracts import OPERATIONAL_V1

    return PolicyObservation(
        time, 1, np.zeros(22), estimate, "observed", OPERATIONAL_V1, estimate.operational.generation
    )


def raw_observation(t=0, frame=1):
    return dict(
        time_s=t,
        frame=frame,
        rgb=np.full((4, 6, 3), 100, np.uint8),
        depth_m=np.ones((4, 6, 1), np.float32),
        valid_depth=np.ones((4, 6, 1), bool),
        joint_position=np.zeros(9),
        joint_velocity=np.zeros(9),
        camera_world=np.eye(4),
        intrinsics=np.eye(3),
    )


def pilot_sensor(t, frame):
    return sample(t, frame) | dict(
        rgb=np.ones((2, 2, 3), np.uint8),
        depth_m=np.ones((2, 2, 1)),
        valid_depth=np.ones((2, 2, 1), bool),
        annotations={"hinge": [9, 9, 9]},
        robot_feedback={"torque": np.ones(7)},
    )


class NumericalEncoder:
    def __init__(self, binding):
        self.binding = binding
        self.reset()

    def reset(self):
        self.calls = []
        self.between = False

    def update(self, value, *, generation):
        self.calls.append(value)
        if self.between:
            return None
        blocks = [
            FeatureBlock(
                np.arange(width), value["time_s"], value["time_s"], value["frame"], generation
            )
            for width in self.binding.config["visual_dims"]
        ]
        return VisualEncoding(*blocks)


def pilot_episode(binding):
    observer = B1Observer(None, binding, encoder=NumericalEncoder(binding))
    observations = [observer.update(pilot_sensor(i / 60, i)) for i in range(3)]
    tool = ObjectFrame(np.zeros(3), np.eye(3))
    return compile_episode(
        episode_id="pilot",
        asset_id="left",
        asset_splits={"left": "train"},
        observations=observations,
        tools=[tool] * 3,
        goals=[tool] * 2,
        joint_targets=np.ones((2, 7)) * 0.01,
        binding=binding,
    )
