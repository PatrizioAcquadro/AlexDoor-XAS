"""Deferred CUDA qualification of the eight B1 model paths (no simulator)."""

import numpy as np
import pytest
import torch

from alexdoor_xas.action.b1 import ACTION_DIMS
from alexdoor_xas.dataset.b1 import export_dataset
from alexdoor_xas.policies.act.config import ActModelCfg
from alexdoor_xas.policies.b1 import (
    B1Policy,
    batch_factories,
    load_b1_data,
    make_model,
    save_policy,
)
from alexdoor_xas.policies.diffusion.config import DiffusionModelCfg
from helpers import TEST_ROBOT_REF, make_b1_episode

pytestmark = pytest.mark.usefixtures("gpu_models")


@pytest.mark.parametrize("family", ["act", "diffusion"])
@pytest.mark.parametrize("space", ACTION_DIMS)
def test_forward_loss_gradients_and_checkpoint_predictions(tmp_path, b1_binding, family, space):
    root = export_dataset(
        [make_b1_episode(b1_binding)],
        tmp_path / "data",
        b1_binding,
        TEST_ROBOT_REF,
        {"left": "train"},
    )
    data = load_b1_data(root, space)
    if family == "act":
        cfg = ActModelCfg(
            chunk_size=4,
            d_model=32,
            n_heads=2,
            dim_feedforward=64,
            z_dim=4,
            cvae_encoder_layers=1,
            encoder_layers=1,
            decoder_layers=1,
            dropout=0,
        )
    else:
        cfg = DiffusionModelCfg(
            horizon=4,
            d_model=32,
            n_heads=2,
            dim_feedforward=64,
            n_decoder_layers=1,
            num_train_timesteps=10,
            dropout=0,
        )
    model = make_model(data, family, cfg)
    batches, _ = batch_factories(data, family, 4, 2, 0)
    raw = next(iter(batches(0)))
    batch = {
        k: torch.as_tensor(v, device="cuda", dtype=torch.bool if k == "is_pad" else torch.float32)
        for k, v in raw.items()
    }
    if family == "act":
        from alexdoor_xas.policies.act.model import act_loss

        predicted, mu, logvar = model(batch["obs"], batch["actions"], batch["is_pad"])
        loss = act_loss(predicted, batch["actions"], batch["is_pad"], mu, logvar, 1)["loss"]
    else:
        from alexdoor_xas.policies.diffusion.model import diffusion_loss
        from alexdoor_xas.policies.diffusion.schedulers import make_train_scheduler

        loss = diffusion_loss(
            model, make_train_scheduler(cfg), batch["actions"], batch["obs"], batch["is_pad"]
        )["loss"]
    assert loss.isfinite()
    loss.backward()
    assert all(p.grad is None or p.grad.isfinite().all() for p in model.parameters())
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())
    path = save_policy(tmp_path / "policy.pt", family, model, data)
    a = B1Policy.from_checkpoint(
        path, family, binding=b1_binding, runtime_asset=TEST_ROBOT_REF, num_inference_steps=5
    )
    b = B1Policy.from_checkpoint(
        path, family, binding=b1_binding, runtime_asset=TEST_ROBOT_REF, num_inference_steps=5
    )
    a.reset(19)
    b.reset(19)
    obs = np.zeros(data.obs_dim)
    np.testing.assert_array_equal(a.predict(obs), b.predict(obs))
