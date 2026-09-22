"""Observed RGB-D samples; no simulator annotations enter this interface."""

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class RGBDSample:
    episode: int
    frame: int
    time_s: float
    rgb: torch.Tensor
    depth_m: torch.Tensor
    valid_depth: torch.Tensor
    joint_position: torch.Tensor
    joint_velocity: torch.Tensor


class RGBDCapture:
    """Copy one fresh rendered frame and matching arm/neck state per sim tick."""

    def __init__(self, near_m: float, far_m: float):
        if not 0 < near_m < far_m:
            raise ValueError("invalid depth interval")
        self.near_m, self.far_m = near_m, far_m
        self.episode = -1
        self.reset()

    def reset(self):
        self.episode += 1
        self.sample = None
        self._last_frame = None

    def capture(self, frame, time_s, rgb, depth, positions, velocities):
        if self._last_frame is not None and frame <= self._last_frame:
            raise ValueError("RGB-D capture requires a fresh camera frame")
        if rgb.shape[-1] not in (3, 4) or depth.shape != (*rgb.shape[:-1], 1):
            raise ValueError("RGB and depth must share the left camera pixel grid")
        if positions.shape[-1] != 9 or velocities.shape != positions.shape:
            raise ValueError("capture requires seven arm and two neck joints")
        if not bool(torch.isfinite(positions).all() and torch.isfinite(velocities).all()):
            raise ValueError("non-finite proprioception")
        valid = torch.isfinite(depth) & (depth > 0) & (depth >= self.near_m)
        valid &= depth <= self.far_m
        self.sample = RGBDSample(
            self.episode,
            int(frame),
            float(time_s),
            rgb[..., :3].clone(),
            depth.clone(),
            valid,
            positions.clone(),
            velocities.clone(),
        )
        self._last_frame = frame
        return self.sample
