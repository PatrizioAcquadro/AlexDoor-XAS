"""ACT numerical model and inference contracts without Isaac imports."""

from __future__ import annotations

import math
from copy import deepcopy

import numpy as np
import pytest
import torch

from alexdoor_xas.policies.act.config import ActModelCfg, ActTrainCfg
from alexdoor_xas.policies.act.model import ACTModel
from alexdoor_xas.policies.act.train import make_seeded_model, train_act

pytestmark = pytest.mark.usefixtures("gpu_models")

TINY_MODEL_CFG = ActModelCfg(
    chunk_size=8,
    d_model=32,
    n_heads=2,
    dim_feedforward=64,
    z_dim=4,
    cvae_encoder_layers=1,
    encoder_layers=1,
    decoder_layers=1,
    dropout=0.0,
)
OBS_DIM = 14
ACTION_DIM = 6


def _tiny_model(seed: int = 0) -> ACTModel:
    torch.manual_seed(seed)
    return ACTModel(obs_dim=OBS_DIM, action_dim=ACTION_DIM, cfg=TINY_MODEL_CFG)


def _tiny_batch(batch: int = 4, seed: int = 0) -> dict[str, torch.Tensor]:
    generator = torch.Generator(device="cuda").manual_seed(seed)
    horizon = TINY_MODEL_CFG.chunk_size
    is_pad = torch.zeros(batch, horizon, dtype=torch.bool)
    is_pad[:, horizon - 2 :] = True
    return {
        "obs": torch.randn(batch, OBS_DIM, generator=generator),
        "actions": torch.randn(batch, horizon, ACTION_DIM, generator=generator),
        "is_pad": is_pad,
    }


def test_forward_rejects_wrong_chunk_length() -> None:
    model = _tiny_model()
    batch = _tiny_batch()
    with pytest.raises(ValueError, match="expected actions of shape"):
        model(batch["obs"], batch["actions"][:, :-1], batch["is_pad"][:, :-1])


def _constant_mapping_batch(batch: int = 8) -> dict[str, np.ndarray]:
    """A deterministic obs -> chunk mapping the tiny model must overfit."""
    horizon = TINY_MODEL_CFG.chunk_size
    obs = np.tile(np.linspace(-1.0, 1.0, OBS_DIM), (batch, 1))
    ramp = np.linspace(-0.5, 0.5, horizon).reshape(1, horizon, 1)
    actions = np.tile(ramp, (batch, 1, ACTION_DIM))
    return {
        "obs": obs,
        "actions": actions,
        "is_pad": np.zeros((batch, horizon), dtype=bool),
    }


def test_train_act_overfits_a_constant_mapping() -> None:
    model = _tiny_model()
    batch = _constant_mapping_batch()
    cfg = ActTrainCfg(
        epochs=200,
        batch_size=8,
        lr=1e-3,
        kl_weight=1.0,
        seed=0,
        val_every=50,
        device="cuda",
    )

    history = train_act(
        model,
        make_train_batches=lambda epoch: [batch],
        cfg=cfg,
        make_val_batches=lambda: [batch],
    )

    assert len(history.epochs) == cfg.epochs
    first, last = history.epochs[0], history.epochs[-1]
    assert last.train_l1 < 0.3 * first.train_l1
    assert last.train_l1 < 0.15
    assert last.val_l1 is not None and math.isfinite(last.val_l1)
    assert 0 <= history.best_epoch < cfg.epochs
    assert history.best_val_l1 <= last.val_l1 + 1e-12


def test_train_act_resume_matches_uninterrupted_state() -> None:
    batch = _constant_mapping_batch(batch=4)
    cfg = ActTrainCfg(
        epochs=4,
        batch_size=4,
        lr=1e-3,
        seed=17,
        val_every=1,
        device="cuda",
    )
    full_model = make_seeded_model(OBS_DIM, ACTION_DIM, TINY_MODEL_CFG, seed=17)
    full_history = train_act(full_model, lambda epoch: [batch], cfg)

    interrupted_model = make_seeded_model(OBS_DIM, ACTION_DIM, TINY_MODEL_CFG, seed=17)
    captured: dict = {}

    class StopAfterEpoch(RuntimeError):
        pass

    def interrupt_after_two_epochs(state) -> None:
        if state["next_epoch"] == 2:
            captured["training_state"] = deepcopy(state)
            captured["model_state"] = {
                key: value.detach().clone() for key, value in interrupted_model.state_dict().items()
            }
            raise StopAfterEpoch

    with pytest.raises(StopAfterEpoch):
        train_act(
            interrupted_model,
            lambda epoch: [batch],
            cfg,
            on_checkpoint=interrupt_after_two_epochs,
        )

    resumed_model = make_seeded_model(OBS_DIM, ACTION_DIM, TINY_MODEL_CFG, seed=17)
    resumed_model.load_state_dict(captured["model_state"])
    resumed_history = train_act(
        resumed_model,
        lambda epoch: [batch],
        cfg,
        resume_state=captured["training_state"],
    )
    for key, value in full_model.state_dict().items():
        torch.testing.assert_close(value, resumed_model.state_dict()[key], rtol=0, atol=0)
    assert [entry.train_loss for entry in resumed_history.epochs] == pytest.approx(
        [entry.train_loss for entry in full_history.epochs]
    )
