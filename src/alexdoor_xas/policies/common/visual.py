"""Compact trainable RGB-D conditioning shared by both policy families."""

import torch
from torch import nn

from alexdoor_xas.policies.encoding import HISTORY, IMAGE_DIM, IMAGE_SIZE, LEARNED_RECIPE


class RGBDProjection(nn.Module):
    """Encode static + oldest-first causal images with one shared convolution.

    Five channels are RGB, metric depth / 5 m and valid-depth coverage. GroupNorm
    uses no running batch statistics, so validation never alters its state.
    Weights are initialized by PyTorch and optimized jointly with the action
    model. A 3x4 spatial grid preserves layout before a 64-dimensional projection.
    Proprioception stays separate until the final observation projection.
    """

    def __init__(self, obs_dim, output_dim):
        super().__init__()
        if obs_dim != (HISTORY + 1) * IMAGE_DIM + 18:
            raise ValueError("Learned RGB-D input width differs from the bound image recipe")
        self.images = nn.Sequential(
            nn.Conv2d(5, 16, 5, stride=2, padding=2),
            nn.GroupNorm(4, 16),
            nn.SiLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.GroupNorm(4, 32),
            nn.SiLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.GroupNorm(4, 64),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d((3, 4)),
            nn.Flatten(),
            nn.Linear(64 * 3 * 4, 64),
            nn.SiLU(),
        )
        self.project = nn.Linear((HISTORY + 1) * 64 + 18, output_dim)

    def forward(self, obs):
        batch = obs.shape[0]
        images = obs[:, :-18].reshape(batch * (HISTORY + 1), 5, *IMAGE_SIZE)
        visual = self.images(images).reshape(batch, -1)
        # Tensor concatenation keeps visual gradients connected to the action loss.
        return self.project(torch.cat((visual, obs[:, -18:]), dim=-1))


def observation_projection(obs_dim, output_dim, recipe=None):
    if recipe is None:
        return nn.Linear(obs_dim, output_dim)
    if recipe != LEARNED_RECIPE:
        raise ValueError("Unsupported trainable visual recipe")
    return RGBDProjection(obs_dim, output_dim)
