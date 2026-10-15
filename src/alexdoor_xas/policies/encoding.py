"""Fixed RGB-D spatial pooling and causal history, shared by preparation and rollout."""

from collections import deque

import numpy as np
import torch
import torch.nn.functional as F

from alexdoor_xas.policies.common.b1_contract import PILOT_SCHEMA, PilotBinding
from alexdoor_xas.policies.observations import FeatureBlock, VisualEncoding

RECIPE = "rgbd-grid-v1"
GRID = 8
HISTORY = 3
PERIOD_S = 0.1
BLOCK_DIM = 5 * GRID * GRID


def pilot_binding(inspection, *, depth_interval_m=(0.1, 5.0), warmup_s=0.2):
    """All preprocessing choices travel with data and checkpoints."""
    return PilotBinding.from_dict(
        dict(
            schema=PILOT_SCHEMA,
            offline_passed=False,
            dynamic_passed=False,
            frozen=False,
            config=dict(
                encoder=RECIPE,
                visual_dims=[BLOCK_DIM, BLOCK_DIM * HISTORY],
                geometry_profile="operational-v1",
                max_gap_s=0.15,
                inspection=inspection,
                warmup_s=warmup_s,
                encoding=dict(
                    grid=GRID,
                    history=HISTORY,
                    period_s=PERIOD_S,
                    depth_interval_m=list(depth_interval_m),
                ),
            ),
        )
    )


class RGBDEncoder:
    """No weights. RGB [0,1], masked metric depth/far, valid coverage, channel-major.

    Static is the last acquired inspection image. Recent history is oldest-first
    at 10 Hz, left-padded at reset. No centered windows, future data or labels.
    In synchronous simulation completion is on the acquisition clock; wall time
    is measured separately and does not renew sensor or geometry support.
    """

    def __init__(self, binding, *, device="cuda:0"):
        expected = dict(grid=GRID, history=HISTORY, period_s=PERIOD_S)
        cfg = binding.config.get("encoding", {})
        interval = cfg.get("depth_interval_m", [])
        if (
            binding.config["encoder"] != RECIPE
            or binding.config["visual_dims"] != [BLOCK_DIM, BLOCK_DIM * HISTORY]
            or any(cfg.get(k) != v for k, v in expected.items())
            or len(interval) != 2
            or not 0 < interval[0] < interval[1]
        ):
            raise ValueError("Unsupported RGB-D encoder recipe")
        self.binding, self.device = binding, torch.device(device)
        self.near, self.far = interval
        self.static_until = binding.config["inspection"]["sample_times_s"][-1]
        self.reset()

    def reset(self):
        self.history = deque(maxlen=HISTORY)
        self.static = self.last = None
        self.generation = None

    @torch.inference_mode()
    def encode(self, sensor):
        def tensor(key):
            return torch.as_tensor(sensor[key], device=self.device)

        rgb = tensor("rgb")
        depth, valid = tensor("depth_m"), tensor("valid_depth")
        if rgb.dtype != torch.uint8 or valid.dtype != torch.bool:
            raise ValueError("RGB-D requires uint8 RGB and boolean validity")
        expected = torch.isfinite(depth) & (depth >= self.near) & (depth <= self.far)
        if not torch.equal(valid, expected):
            raise ValueError("Depth mask differs from the bound sensor interval")
        rgb = rgb.permute(2, 0, 1).float()[None] / 255
        mask = valid.permute(2, 0, 1).float()[None]
        depth = torch.where(valid, depth, 0).permute(2, 0, 1).float()[None] / self.far

        def pool(x):
            return F.adaptive_avg_pool2d(x, (GRID, GRID))

        coverage = pool(mask)
        depth = pool(depth) / coverage.clamp_min(1e-8)
        return torch.cat((pool(rgb), depth, coverage), dim=1).flatten().cpu().numpy()

    def update(self, sensor, *, generation):
        if generation != self.generation:
            self.reset()
            self.generation = generation
        t, frame = float(sensor["time_s"]), int(sensor["frame"])
        if self.last is not None and t - self.last.recent.acquired_s < PERIOD_S - 1e-9:
            return None
        vector = self.encode(sensor)
        block = FeatureBlock(vector, t, t, frame, generation)
        if self.static is None or t <= self.static_until + 1e-9:
            self.static = block
        self.history.append(vector)
        padded = [self.history[0]] * (HISTORY - len(self.history)) + list(self.history)
        recent = FeatureBlock(np.concatenate(padded), t, t, frame, generation)
        self.last = VisualEncoding(self.static, recent)
        return self.last
