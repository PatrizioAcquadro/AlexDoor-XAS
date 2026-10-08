"""Diffusion normalization and scheduler configuration without model execution."""

from __future__ import annotations

import numpy as np
import pytest

from alexdoor_xas.policies.diffusion.config import (
    DiffusionConfigError,
    DiffusionModelCfg,
)
from alexdoor_xas.policies.diffusion.data import (
    MinMaxNormalizer,
    make_diffusion_normalizer,
)
from alexdoor_xas.policies.diffusion.schedulers import (
    make_inference_scheduler,
)
from helpers import diffusion_action_stats, diffusion_stats

ACTION_DIM = 6


OBS_DIM = 14


TINY_MODEL_CFG = DiffusionModelCfg(
    horizon=8,
    d_model=32,
    n_heads=2,
    n_decoder_layers=1,
    dim_feedforward=64,
    dropout=0.0,
    num_train_timesteps=25,
)


def test_minmax_round_trip_and_extrema() -> None:
    stats = diffusion_action_stats()
    normalizer = MinMaxNormalizer.from_norm_stats(stats)

    rng = np.random.default_rng(0)
    x = rng.uniform(stats.min, stats.max, size=(50, ACTION_DIM))
    np.testing.assert_allclose(normalizer.denormalize(normalizer.normalize(x)), x, atol=1e-12)

    # Train extrema map to exactly ±1 on non-constant dims.
    np.testing.assert_allclose(normalizer.normalize(stats.min)[:3], -1.0)
    np.testing.assert_allclose(normalizer.normalize(stats.max)[:3], 1.0)
    # Everything inside the train range stays within [-1, 1].
    assert np.abs(normalizer.normalize(x)).max() <= 1.0 + 1e-12


def test_minmax_constant_dims_shift_to_exact_zero() -> None:
    normalizer = MinMaxNormalizer.from_norm_stats(diffusion_action_stats())

    zeros = np.zeros(ACTION_DIM)
    assert (normalizer.normalize(zeros)[3:] == 0.0).all()
    # A sampled (clipped) normalized value denormalizes through scale 1.0 —
    # no division by a floored std, no 1e8 blowup.
    sampled = np.array([0.5, -0.5, 0.25, 0.9, -0.9, 0.1])
    denorm = normalizer.denormalize(sampled)
    assert np.abs(denorm[3:]).max() <= 1.0  # bounded, not exploded
    assert np.isfinite(denorm).all()


def test_diffusion_batch_normalizer_mixes_zscore_obs_and_minmax_actions() -> None:
    stats = diffusion_stats()
    normalize = make_diffusion_normalizer(stats)
    batch = {
        "obs": np.full((4, OBS_DIM), 0.5),
        "actions": np.tile(stats.action.max, (4, TINY_MODEL_CFG.horizon, 1)),
        "is_pad": np.zeros((4, TINY_MODEL_CFG.horizon), dtype=bool),
    }

    out = normalize(batch, stats)

    np.testing.assert_allclose(out["obs"], 1.0)  # (0.5 - 0) / 0.5
    np.testing.assert_allclose(out["actions"][..., :3], 1.0)
    np.testing.assert_allclose(out["actions"][..., 3:], 0.0)
    assert out["is_pad"] is batch["is_pad"]  # passthrough


def test_inference_scheduler_validates_inputs() -> None:
    with pytest.raises(DiffusionConfigError, match="num_inference_steps"):
        make_inference_scheduler(TINY_MODEL_CFG, "ddpm", 26)
    with pytest.raises(DiffusionConfigError, match="sampler"):
        make_inference_scheduler(TINY_MODEL_CFG, "heun", 10)
