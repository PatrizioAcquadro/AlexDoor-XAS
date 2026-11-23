"""Shared flat/temporal batch transfer and retained scientific-reader imports."""

import numpy as np
import pytest
import torch

from alexdoor_xas.policies.act.train import _to_device as act_to_device
from alexdoor_xas.policies.common.data import to_device
from alexdoor_xas.policies.diffusion.train import _to_device as diffusion_to_device


@pytest.mark.parametrize("device", ["cpu", "meta"])
def test_transfer_preserves_tensor_shapes_and_non_tensor_metadata(device):
    metadata = np.array([2, 3])
    batch = dict(
        obs=dict(images=torch.ones(2, 5, 4, 4), robot=torch.arange(6).reshape(2, 3)),
        actions=torch.zeros(2, 3),
        is_pad=torch.tensor([False, True]),
        source=metadata,
        label="recorded",
    )
    moved = to_device(batch, torch.device(device))
    assert act_to_device is diffusion_to_device is to_device
    for before, after in [(batch, moved), (batch["obs"], moved["obs"])]:
        assert before is not after
        for key, value in before.items():
            if isinstance(value, torch.Tensor):
                assert after[key].device.type == device
                assert (after[key].shape, after[key].dtype) == (value.shape, value.dtype)
                if device == "cpu":
                    torch.testing.assert_close(after[key], value)
    assert moved["source"] is metadata and moved["label"] == "recorded"
    assert batch["obs"]["images"].device.type == "cpu"
