"""Shared observed-only static scan fixtures; no model execution."""

from pathlib import Path

import numpy as np

from alexdoor_xas.perception.provider import load_recipe


def sensor(t=4.0, frame=248):
    return dict(
        time_s=t,
        frame=frame,
        rgb=np.ones((16, 16, 3), np.uint8),
        depth_m=np.ones((16, 16, 1)),
        valid_depth=np.ones((16, 16, 1), bool),
        joint_position=np.zeros(9),
        joint_velocity=np.zeros(9),
        camera_world=np.eye(4),
        intrinsics=np.array([[20, 0, 8], [0, 20, 8], [0, 0, 1]]),
    )


class EmptyWorker:
    def infer(self, rgb):
        return dict(
            latency_s=0.05,
            shape=rgb.shape[:2],
            masks=[],
            tokens=bytes(196 * 384 * 4),
            token_shape=(196, 384),
            pixel_mapping=dict(scale=14, pad_x=0, pad_y=0),
        )


def recipe():
    root = Path(__file__).resolve().parents[1]
    return load_recipe(root / "configs/perception_geometry.json", root)
