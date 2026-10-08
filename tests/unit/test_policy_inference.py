"""Inference adapters and loss arithmetic with stub models; no neural execution."""

import math
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from alexdoor_xas.policies.act.config import ActModelCfg
from alexdoor_xas.policies.act.model import act_loss
from alexdoor_xas.policies.act.policy import ActPolicy, act_chunk_source
from alexdoor_xas.policies.diffusion.config import DiffusionModelCfg
from alexdoor_xas.policies.diffusion.policy import DiffusionPolicy, diffusion_chunk_source
from alexdoor_xas.policies.diffusion.schedulers import make_inference_scheduler, sample_actions
from helpers import diffusion_stats

OBS_DIM = 14
ACTION_DIM = 6


class CaptureModel(torch.nn.Module):
    obs_dim = OBS_DIM
    action_dim = ACTION_DIM

    def __init__(self, family):
        super().__init__()
        self.cfg = (
            ActModelCfg(chunk_size=8)
            if family == "act"
            else DiffusionModelCfg(horizon=8, num_train_timesteps=25)
        )
        self.last_input = None

    def predict(self, obs):
        self.last_input = obs.detach().clone()
        return torch.ones(obs.shape[0], self.cfg.chunk_size, self.action_dim)

    def forward(self, actions, timestep, obs):
        self.last_input = obs.detach().clone()
        return torch.zeros_like(actions)


def policy(family, model, stats):
    if family == "act":
        return ActPolicy(model, stats)
    return DiffusionPolicy(model, stats, num_inference_steps=2)


@pytest.mark.parametrize("family", ["act", "diffusion"])
def test_policy_normalizes_observations_and_clips_extremes(family):
    stats = diffusion_stats()
    stats.obs.mean[:] = np.linspace(1.0, 2.0, OBS_DIM)
    model = CaptureModel(family)
    inference = policy(family, model, stats)
    chunk = inference.predict(stats.obs.mean)
    np.testing.assert_array_equal(model.last_input.numpy(), np.zeros((1, OBS_DIM)))
    assert chunk.shape == (8, ACTION_DIM)
    if family == "act":
        np.testing.assert_allclose(chunk, np.tile(stats.action.mean + stats.action.std, (8, 1)))
    inference.predict(np.full(OBS_DIM, 1e6))
    np.testing.assert_array_equal(
        model.last_input.numpy(), np.full((1, OBS_DIM), inference.obs_clip)
    )


@pytest.mark.parametrize("family", ["act", "diffusion"])
def test_policy_rejects_mismatched_statistics(family):
    model = CaptureModel(family)
    model.obs_dim += 1
    with pytest.raises(ValueError, match="obs dim"):
        policy(family, model, diffusion_stats())


def test_chunk_sources_use_observations_and_requested_horizon():
    stub = SimpleNamespace(chunk_size=8, predict=lambda obs: np.arange(56).reshape(8, 7) + obs[0])
    for factory, options, steps in (
        (act_chunk_source, {}, 8),
        (diffusion_chunk_source, {}, 8),
        (diffusion_chunk_source, {"n_action_steps": 3}, 3),
    ):
        emit = factory(stub, lambda context: context, **options)
        for obs in (np.array([1.0]), np.array([2.0])):
            np.testing.assert_array_equal(emit(obs), stub.predict(obs)[:steps])
    with pytest.raises(ValueError, match="n_action_steps"):
        diffusion_chunk_source(stub, lambda context: context, n_action_steps=9)


def test_temporal_ensemble_weights_match_the_paper_scheme() -> None:
    m = 0.5
    chunk_a = np.tile(np.array([[1.0, 0, 0, 0, 0, 0]]), (3, 1)) * np.array([[1], [2], [3]])
    chunk_b = np.tile(np.array([[10.0, 0, 0, 0, 0, 0]]), (3, 1))
    chunks = iter([chunk_a, chunk_b])
    source = act_chunk_source(
        SimpleNamespace(predict=lambda obs: next(chunks)),
        lambda ctx: np.zeros(OBS_DIM),
        temporal_ensemble=True,
        ensemble_m=m,
    )
    ctx = object()

    first = source(ctx)
    assert first.shape == (1, 6)
    np.testing.assert_allclose(first[0], chunk_a[0])

    second = source(ctx)
    weights = np.array([1.0, math.exp(-m)])  # oldest chunk first, weight exp(-m * i)
    expected = (chunk_a[1] * weights[0] + chunk_b[0] * weights[1]) / weights.sum()
    np.testing.assert_allclose(second[0], expected)


def _loss_batch():
    actions = torch.arange(4 * 8 * ACTION_DIM, dtype=torch.float32).reshape(4, 8, ACTION_DIM) / 100
    is_pad = torch.zeros(4, 8, dtype=torch.bool)
    is_pad[:, -2:] = True
    return {"actions": actions, "is_pad": is_pad}


def test_masked_l1_ignores_padded_slots() -> None:
    batch = _loss_batch()
    a_hat = torch.zeros_like(batch["actions"])
    mu = torch.zeros(4, 4)
    logvar = torch.zeros(4, 4)

    base = act_loss(a_hat, batch["actions"], batch["is_pad"], mu, logvar, kl_weight=1.0)
    corrupted = batch["actions"].clone()
    corrupted[batch["is_pad"]] = 1e6
    altered = act_loss(a_hat, corrupted, batch["is_pad"], mu, logvar, kl_weight=1.0)

    assert base["l1"].item() == pytest.approx(altered["l1"].item())
    assert base["loss"].item() == pytest.approx(altered["loss"].item())


def test_kl_term_is_zero_at_prior_and_positive_away_from_it() -> None:
    batch = _loss_batch()
    a_hat = batch["actions"].clone()
    zeros = torch.zeros(4, 4)

    at_prior = act_loss(a_hat, batch["actions"], batch["is_pad"], zeros, zeros, kl_weight=1.0)
    assert at_prior["kl"].item() == pytest.approx(0.0)
    assert at_prior["l1"].item() == pytest.approx(0.0)

    off_prior = act_loss(
        a_hat, batch["actions"], batch["is_pad"], zeros + 2.0, zeros, kl_weight=1.0
    )
    assert off_prior["kl"].item() > 0.0
    assert off_prior["loss"].item() == pytest.approx(off_prior["kl"].item())


def test_all_padded_batch_is_rejected() -> None:
    batch = _loss_batch()
    zeros = torch.zeros(4, 4)
    with pytest.raises(ValueError, match="all-padded"):
        act_loss(
            batch["actions"],
            batch["actions"],
            torch.ones_like(batch["is_pad"]),
            zeros,
            zeros,
            kl_weight=1.0,
        )


def test_diffusion_samplers_repeat_seeded_noise_and_bound_actions():
    model = CaptureModel("diffusion")
    obs = torch.zeros(3, OBS_DIM)

    def sample(sampler, seed):
        scheduler = make_inference_scheduler(model.cfg, sampler, 10)
        return sample_actions(
            model, scheduler, obs, 8, ACTION_DIM, torch.Generator().manual_seed(seed)
        )

    for sampler in ("ddpm", "ddim"):
        first = sample(sampler, 7)
        assert first.shape == (3, 8, ACTION_DIM)
        assert torch.isfinite(first).all()
        assert first.abs().max() <= 1.0 + 1e-6
        torch.testing.assert_close(first, sample(sampler, 7), rtol=0, atol=0)
        assert not torch.allclose(first, sample(sampler, 8))
