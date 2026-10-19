"""CUDA gradients, masks, reuse and checkpoint contracts; no optimizer updates."""

from dataclasses import replace

import h5py
import numpy as np
import pytest
import torch

from alexdoor_xas.dataset.sampling import collate_torch
from alexdoor_xas.dataset.temporal import TemporalEpisode, prepare_temporal
from alexdoor_xas.policies.act.model import act_loss
from alexdoor_xas.policies.act.temporal import (
    TemporalActPolicy,
    fit_temporal_stats,
    normalize_temporal_batch,
    save_temporal_policy,
    temporal_model,
)
from alexdoor_xas.policies.act.train import _to_device
from alexdoor_xas.policies.temporal import TemporalConfig
from helpers import TEST_ROBOT_REF
from helpers import temporal_command as record_command
from helpers import temporal_recorder as recorder


def prepare_fixture(tmp_path):
    path = tmp_path / "raw.hdf5"
    log = recorder(path)
    log.reset(0)
    for tick in range(31):
        t = tick / 60
        log.state(np.zeros(9), np.zeros(9), acquired=t, available=t, generation=0)
        if tick % 6 == 0:
            depth = np.ones((60, 96, 1), "f4")
            log.capture(
                np.full((60, 96, 3), tick + 50, "u1"),
                depth,
                np.ones_like(depth, bool),
                np.eye(3),
                np.eye(4),
                acquired=t,
                available=t,
                generation=0,
            )
        log.feedback(
            np.full(7, 0.1 + tick / 100), acquired=t, available=t, healthy=True, generation=0
        )
        if tick < 30:
            record_command(log, t)
    log.finish(dict(passed=False, termination="fixture_not_physical_success"))
    log.close()
    cfg = TemporalConfig(history_s=0.4, image_size=(60, 96), static_time_s=0.1, warmup_s=0.1)
    eligible = np.ones(30, bool)
    eligible[20] = False  # ambiguous real action cannot be relabeled padding
    prepare_temporal(path, tmp_path / "frames.hdf5", cfg, eligible=eligible)
    return cfg, TemporalEpisode(tmp_path / "frames.hdf5", cfg)


def test_preparation_sampling_horizons_padding_and_train_only_stats(tmp_path):
    assert torch.cuda.is_available(), "GPU required, do not fall back to CPU"
    cfg, ep = prepare_fixture(tmp_path)
    assert not ep.sample(20)["supervision_eligible"]
    assert not ep.sample(12)["supervision_eligible"]  # crosses ambiguous row
    assert ep.sample(25)["supervision_eligible"]
    tail = ep.sample(29)
    assert tail["is_pad"].sum() == 11 and not tail["is_pad"][0]
    terminal = ep.sample(30)
    assert terminal["is_pad"].all() and not terminal["supervision_eligible"]
    assert not ep.sample(0)["ready"]
    ep.frames.cache.clear()
    before_reads = ep.frames.reads
    rows = ep.training_rows()
    assert ep.frames.reads == before_reads
    expected = [row for row in range(len(ep.times)) if ep.sample(row)["supervision_eligible"]]
    np.testing.assert_array_equal(rows, expected)
    assert len(list(ep.batches(rows, batch_size=4))) > 0
    size = ep.path.stat().st_size
    for history in (0.2, 0.4, 0.6):
        view = TemporalEpisode(ep.path, replace(cfg, history_s=history, image_offsets_s=None))
        assert view.sample(25, 64)["actions"].shape == (64, 7)
        view.close()
    sparse = TemporalEpisode(ep.path, replace(cfg, state_hz=30))
    np.testing.assert_array_equal(sparse.sample(29)["actions"], ep.sample(29)["actions"])
    np.testing.assert_array_equal(sparse.sample(29)["is_pad"], ep.sample(29)["is_pad"])
    sparse.close()
    with pytest.raises(ValueError, match="control rate"):
        TemporalEpisode(ep.path, replace(cfg, control_dt_s=1 / 30))
    assert ep.path.stat().st_size == size
    with h5py.File(ep.path) as file:
        assert len(file["images/values"]) == 6
    stats = fit_temporal_stats([ep])
    assert stats.obs.dim == 32 and stats.action.dim == 7
    assert stats.obs.mean[18] != 0
    ep.metadata["split"] = "development"
    with pytest.raises(ValueError, match="training"):
        fit_temporal_stats([ep])
    ep.close()


def test_gradients_masks_cache_invalidation_and_checkpoint(tmp_path):
    assert torch.cuda.is_available(), "GPU required, do not fall back to CPU"
    cfg, ep = prepare_fixture(tmp_path)
    stats = fit_temporal_stats([ep])
    model = temporal_model(
        cfg,
        d_model=32,
        n_heads=2,
        dim_feedforward=64,
        cvae_encoder_layers=1,
        encoder_layers=1,
        decoder_layers=1,
        dropout=0,
    ).cuda()
    batch = _to_device(collate_torch(normalize_temporal_batch(ep.batch([25, 26]), stats)), "cuda")
    model.train()
    batch["obs"]["robot"].requires_grad_(True)
    before = {key: value.detach().clone() for key, value in model.state_dict().items()}
    predicted, mu, logvar = model(batch["obs"], batch["actions"], batch["is_pad"])
    act_loss(predicted, batch["actions"], batch["is_pad"], mu, logvar, 1)["loss"].backward()
    for module in (model.obs_proj.images, model.obs_proj.temporal):
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in module.parameters())
        assert all(p.grad is None or p.grad.isfinite().all() for p in module.parameters())
    assert not model.obs_proj.cache
    assert batch["obs"]["robot"].grad[:, -1, :18].abs().sum() > 0
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, before[key], rtol=0, atol=0)
    policy = TemporalActPolicy(model, stats)
    policy.reset(ep.builder.episode_id, 0)
    snap = ep.sample(25)
    full = policy.predict(snap, reuse=False)
    cached = policy.predict(snap, reuse=True)
    np.testing.assert_allclose(full, cached, atol=1e-6, rtol=1e-5)
    again = policy.predict(snap, reuse=True)
    np.testing.assert_array_equal(again, cached)
    assert model.obs_proj.cache_hits > 0
    with torch.no_grad():
        next(model.obs_proj.images.parameters()).add_(0.01)
    changed = policy.predict(snap, reuse=True)
    np.testing.assert_allclose(changed, policy.predict(snap, reuse=False), atol=1e-6, rtol=1e-5)
    # Masked values cannot affect predictions; trained weights are unchanged by checks.
    raw = {key: value.copy() for key, value in snap["obs"].items()}
    raw["robot_valid"][:, 2] = False
    altered = dict(snap, obs=raw)
    masked = policy.predict(altered, reuse=False)
    raw["robot"][:, 18:25] = 1e6
    np.testing.assert_array_equal(masked, policy.predict(altered, reuse=False))
    path = save_temporal_policy(tmp_path / "new.pt", model, stats, TEST_ROBOT_REF)
    loaded = TemporalActPolicy.from_checkpoint(path, config=cfg, runtime_asset=TEST_ROBOT_REF)
    loaded.reset(ep.builder.episode_id, 0)
    np.testing.assert_allclose(loaded.predict(snap), changed, atol=1e-6, rtol=1e-5)
    # Same initialized/saved checkpoint, different real acquisition/warmup metadata.
    intermediate_cfg = replace(cfg, static_time_s=0, warmup_s=0.2)
    intermediate = TemporalActPolicy.from_checkpoint(
        path, config=intermediate_cfg, runtime_asset=TEST_ROBOT_REF
    )
    intermediate.reset(ep.builder.episode_id, 0)
    from alexdoor_xas.policies.temporal import CausalObservationBuilder
    intermediate_snap = CausalObservationBuilder(
        intermediate_cfg, ep.builder.streams, episode_id=ep.builder.episode_id
    ).build(ep.times[25], 0)
    assert np.isfinite(intermediate.predict(intermediate_snap)).all()
    loaded.reset(ep.builder.episode_id, 0)
    np.testing.assert_array_equal(
        intermediate.predict(intermediate_snap), loaded.predict(intermediate_snap)
    )
    with pytest.raises(ValueError, match="mismatch"):
        TemporalActPolicy.from_checkpoint(
            path, config=replace(cfg, image_size=(120, 192)), runtime_asset=TEST_ROBOT_REF
        )
    # Same dimensions with a changed preprocessing recipe are incompatible.
    changed_recipe = dict(snap, observation_config=replace(cfg, depth_scale_m=6).to_dict())
    with pytest.raises(ValueError, match="preprocessing"):
        policy.predict(changed_recipe, reuse=True)
    with pytest.raises(ValueError, match="legacy"):
        policy.predict(np.zeros(32))
    with pytest.raises(ValueError, match="stale"):
        policy.predict(snap, now_s=snap["decision_time_s"] + 1)
    policy.model.load_state_dict(policy.model.state_dict())
    np.testing.assert_allclose(policy.predict(snap, reuse=True), changed, atol=1e-6, rtol=1e-5)
    policy.model.train()
    assert not policy.model.obs_proj.cache
    policy.model.eval()
    policy.reset(ep.builder.episode_id, 1)
    assert not model.obs_proj.cache
    with pytest.raises(ValueError, match="Reset"):
        policy.predict(snap)
    ep.close()
