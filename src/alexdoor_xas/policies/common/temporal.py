"""Minimal finite-history conditioning using the maintained RGB-D convolution."""

from collections import OrderedDict

import torch
from torch import nn

from alexdoor_xas.policies.common.visual import RGBDProjection
from alexdoor_xas.policies.encoding import HISTORY, IMAGE_DIM
from alexdoor_xas.policies.temporal import TemporalConfig


class TemporalProjection(nn.Module):
    """Shared CNN + one finite attention block over image and measured-state tokens.

    Masks remove missing tokens. Relative ages and availability delays describe
    irregular support. Every call recomputes temporal context. Inference reuse is
    explicit, bounded and bypassed whenever gradients or training are enabled.
    """

    def __init__(self, output_dim, config, *, n_heads=4, dropout=0.1):
        super().__init__()
        self.config = TemporalConfig(**config)
        # Reuse the exact existing visual encoder, including its 3x4 spatial grid.
        self.images = RGBDProjection((HISTORY + 1) * IMAGE_DIM + 18, output_dim).images
        self.visual_token = nn.Linear(64 + 2, output_dim)
        self.robot_token = nn.Linear(32 + 4 + 8, output_dim)
        self.summary = nn.Parameter(torch.zeros(1, 1, output_dim))
        self.kind = nn.Parameter(torch.zeros(2, output_dim))
        self.temporal = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                output_dim,
                n_heads,
                output_dim * 4,
                dropout=dropout,
                batch_first=True,
            ),
            num_layers=1,
            enable_nested_tensor=False,
        )
        self.readout = nn.Linear(output_dim + 18, output_dim)
        self.cache = OrderedDict()
        self.cache_identity = None
        self.cache_capacity = 64
        self.cache_hits = self.cache_misses = 0

    def clear_cache(self):
        self.cache.clear()
        self.cache_identity = None

    def train(self, mode=True):
        self.clear_cache()
        return super().train(mode)

    def _apply(self, fn, recurse=True):
        self.clear_cache()
        return super()._apply(fn, recurse=recurse)

    def visual(self, obs):
        images = obs["images"]
        batch, count = images.shape[:2]
        keys = obs.get("cache_keys")
        use_cache = (
            not self.training and not torch.is_grad_enabled() and keys is not None and batch == 1
        )
        if not use_cache:
            return self.images(images.reshape(batch * count, *images.shape[2:])).reshape(
                batch, count, 64
            )
        # Tensor versions catch in-place weight edits and load_state_dict. Device,
        # dtype and complete preprocessing recipe invalidate too. Keys include
        # physical episode, generation and source index, supplied by the policy.
        encoder_device = next(self.images.parameters()).device
        identity = (
            tuple((p.data_ptr(), p._version) for p in self.images.parameters()),
            str(encoder_device),
            images.dtype,
            repr(self.config),
        )
        if identity != self.cache_identity:
            self.clear_cache()
            self.cache_identity = identity
        missing = list(dict.fromkeys(key for key in keys if key not in self.cache))
        if missing:
            positions = [keys.index(key) for key in missing]
            features = self.images(images[0, positions].to(encoder_device))
            for key, feature in zip(missing, features, strict=True):
                self.cache[key] = feature.detach().clone()
                self.cache_misses += 1
        result = torch.stack([self.cache[key] for key in keys])[None]
        for key in keys:
            self.cache.move_to_end(key)
        while len(self.cache) > self.cache_capacity:
            self.cache.popitem(last=False)
        self.cache_hits += len(keys) - len(missing)
        return result

    def forward(self, obs):
        cfg = self.config
        if (
            not isinstance(obs, dict)
            or obs["images"].shape[1:] != (cfg.image_count, 5, *cfg.image_size)
            or obs["robot"].shape[1:] != (cfg.state_count, 32)
        ):
            raise ValueError("Temporal observation does not match the checkpoint recipe")
        image_mask, robot_mask = obs["image_valid"].bool(), obs["robot_valid"].bool()
        robot = obs["robot"]
        expanded = torch.cat(
            [
                robot_mask[..., i : i + 1].expand(*robot_mask.shape[:2], width)
                for i, width in enumerate((9, 9, 7, 7))
            ],
            dim=-1,
        )
        robot = torch.where(expanded, robot, 0)
        visual = (
            self.visual_token(torch.cat((self.visual(obs), obs["image_timing"]), dim=-1))
            + self.kind[0]
        )
        measured = (
            self.robot_token(
                torch.cat(
                    (
                        robot,
                        robot_mask.to(robot.dtype),
                        torch.where(robot_mask[..., None], obs["robot_timing"], 0).flatten(-2),
                    ),
                    dim=-1,
                )
            )
            + self.kind[1]
        )
        summary = self.summary.expand(robot.shape[0], -1, -1)
        sequence = torch.cat((summary, visual, measured), dim=1)
        valid = torch.cat(
            (
                torch.ones(robot.shape[0], 1, device=robot.device, dtype=torch.bool),
                image_mask,
                robot_mask.any(-1),
            ),
            dim=1,
        )
        context = self.temporal(sequence, src_key_padding_mask=~valid)
        # The current measured-state token is the readout; current q/dq also
        # remain explicit, as required by the causal observation contract.
        return self.readout(torch.cat((context[:, -1], robot[:, -1, :18]), dim=-1))
