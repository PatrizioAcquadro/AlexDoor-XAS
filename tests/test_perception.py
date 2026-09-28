"""Observed-only geometry, causal loss handling and frozen-backbone behavior."""

import json
from pathlib import Path

import pytest
import torch

from alexdoor_xas.perception.model import (
    DoorEstimate,
    DoorEstimator,
    ObservedEstimator,
    decode,
    estimator_loss,
    preprocess,
)

CONFIG = json.loads((Path(__file__).resolve().parents[1] / "configs/perception.json").read_text())


def test_letterbox_depth_and_intrinsics(gpu_models):
    rgb = torch.full((1, 60, 100, 3), 128, dtype=torch.uint8)
    depth = torch.ones(1, 60, 100, 1)
    depth[:, 0] = float("nan")
    valid = torch.isfinite(depth)
    k = torch.tensor([[[100.0, 0, 50], [0, 100, 30], [0, 0, 1]]])
    pixels, geometry, calibrated = preprocess(rgb, depth, valid, k, 224)
    assert pixels.shape == (1, 3, 224, 224)
    assert torch.isfinite(geometry).all()
    assert not geometry[:, :, :45].any()
    assert float(calibrated[0, 0, 2]) == pytest.approx(112)
    assert float(calibrated[0, 1, 2]) == pytest.approx(112)
    assert not geometry[:, :3][geometry[:, 3:4].expand(-1, 3, -1, -1) == 0].any()


def test_geometry_decode_and_loss_are_finite_without_weight_update(gpu_models):
    model = DoorEstimator().eval()
    before = {k: v.clone() for k, v in model.state_dict().items()}
    with torch.no_grad():
        pred = model(
            torch.randn(2, 4, 384, 16, 16),
            torch.ones(2, 4, 4, 16, 16),
            torch.zeros(2, 4, 18),
            torch.eye(4).repeat(2, 4, 1, 1),
        )
        loss, _ = estimator_loss(pred, {k: v.clone() for k, v in pred.items()})
    assert torch.isfinite(loss)
    assert all(torch.equal(before[k], v) for k, v in model.state_dict().items())
    torch.testing.assert_close(
        pred["hinge_rotation"].transpose(-1, -2) @ pred["hinge_rotation"],
        torch.eye(3).repeat(2, 1, 1),
        atol=1e-5,
        rtol=1e-5,
    )


def test_inference_resets_on_missing_stale_frames_and_requires_history(gpu_models):
    class Backbone(torch.nn.Module):
        def forward(self, rgb):
            return torch.zeros(len(rgb), 384, 16, 16)

    class Head(torch.nn.Module):
        def forward(self, *args):
            raw = torch.zeros(1, 24)
            raw[:, [3, 7, 10, 17, 21]] = 1
            raw[:, 23] = 10
            return decode(raw)

    runner = ObservedEstimator(Backbone(), Head(), CONFIG)
    observation = dict(
        time_s=0.0,
        frame=1,
        rgb=torch.full((60, 100, 3), 128, dtype=torch.uint8),
        depth_m=torch.ones(60, 100, 1),
        valid_depth=torch.ones(60, 100, 1, dtype=torch.bool),
        joint_position=torch.zeros(9),
        joint_velocity=torch.zeros(9),
        intrinsics=torch.tensor([[100.0, 0, 50], [0, 100, 30], [0, 0, 1]]),
        camera_world=torch.eye(4),
    )
    for i in range(4):
        observation.update(time_s=i / 10, frame=i + 1)
        result = runner.update(observation)
        assert result.valid == (i == 3)
    assert result.fresh(0.4)
    assert not result.fresh(0.5)
    assert runner.update(observation).reason == "nonmonotonic_observation"
    observation.update(time_s=0.6, frame=8)
    assert runner.update(observation).reason == "warming_up"
    observation["valid_depth"][:] = False
    observation.update(time_s=0.7, frame=9)
    assert runner.update(observation).reason == "missing_depth"
    assert len(runner.history) == 0
    assert not DoorEstimate(0.0, False, "lost").fresh(0.0)


def test_signed_articulation_reconstructs_both_handed_contacts(gpu_models):
    raw = torch.zeros(2, 24)
    raw[:, :3] = torch.tensor([0.1, 0.2, 0.0])
    raw[:, [3, 7, 17, 21]] = 1
    raw[:, 9] = torch.tensor([1.0, -1.0])
    raw[:, 14:17] = torch.tensor([0.0, 1.0, 1.0])
    predicted = decode(raw)
    torch.testing.assert_close(
        predicted["contact_position"],
        torch.tensor([[-0.9, 0.2, 1.0], [1.1, 0.2, 1.0]]),
        rtol=1e-5,
        atol=1e-6,
    )
