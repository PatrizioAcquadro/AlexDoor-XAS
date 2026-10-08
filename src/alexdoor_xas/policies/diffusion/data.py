"""Z-score observations and scale Diffusion actions to [-1, 1]."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from alexdoor_xas.dataset.normalize import DatasetNormStats, NormStats

RANGE_EPS = 1e-8


@dataclass(frozen=True)
class MinMaxNormalizer:
    """Per-dimension min-max scaling to [-1, 1] with a constant-dim guard."""

    center: np.ndarray  # (D,)
    scale: np.ndarray  # (D,); 1.0 on constant dims (shift-to-zero, no scaling)

    @classmethod
    def from_norm_stats(cls, stats: NormStats, range_eps: float = RANGE_EPS) -> MinMaxNormalizer:
        low = np.asarray(stats.min, dtype=np.float64)
        high = np.asarray(stats.max, dtype=np.float64)
        span = high - low
        constant = span < range_eps
        center = np.where(constant, low, (low + high) / 2.0)
        scale = np.where(constant, 1.0, 2.0 / np.where(constant, 1.0, span))
        return cls(center=center, scale=scale)

    @property
    def dim(self) -> int:
        return int(self.center.shape[0])

    def normalize(self, x: np.ndarray) -> np.ndarray:
        return (np.asarray(x, dtype=np.float64) - self.center) * self.scale

    def denormalize(self, x: np.ndarray) -> np.ndarray:
        return np.asarray(x, dtype=np.float64) / self.scale + self.center


def make_diffusion_normalizer(stats: DatasetNormStats):
    """Batch normalizer: z-score obs (ACT-identical), min-max actions."""
    action_minmax = MinMaxNormalizer.from_norm_stats(stats.action)

    def normalize(batch: dict[str, Any], batch_stats: DatasetNormStats) -> dict[str, Any]:
        normalized = dict(batch)
        normalized["obs"] = batch_stats.obs.normalize(batch["obs"])
        normalized["actions"] = action_minmax.normalize(batch["actions"])
        return normalized

    return normalize
