"""Metric anchoring, causal inspection and qualified inference regressions."""

import json
from pathlib import Path

import pytest
import torch

from alexdoor_xas.perception.inspection import load_inspection
from alexdoor_xas.perception.model import (
    ObservedEstimator,
    make_estimator,
    metric_geometry,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = json.loads((ROOT / "configs/perception_metric.json").read_text())


def test_common_scan_has_physical_limits_and_consistent_model_contract():
    config = load_inspection(ROOT / "configs/perception_inspection.json")
    assert config == CONFIG["inspection"]
    assert config["waypoints"][0][1:] == config["waypoints"][-1][1:]


def test_policy_encoding_preserves_predictions_and_weights(gpu_models):
    model = make_estimator(CONFIG).eval()
    before = {k: v.clone() for k, v in model.state_dict().items()}
    t = model.context_size + CONFIG["history"]
    features = torch.randn(1, t, 384, 16, 16)
    geometry = torch.rand(1, t, 4, 32, 32)
    geometry[:, :, 3] = 1
    proprio = torch.zeros(1, t, 18)
    camera = torch.eye(4).repeat(1, t, 1, 1)
    with torch.no_grad():
        expected = model(features, geometry, proprio, camera)
        actual, encoding = model(features, geometry, proprio, camera, return_encoding=True)
    for key in expected:
        torch.testing.assert_close(actual[key], expected[key], atol=0, rtol=0)
    assert encoding.shape == (1, 2 * CONFIG["hidden_size"])
    assert encoding.isfinite().all()
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, before[key], atol=0, rtol=0)


def test_metric_anchor_moves_with_world_and_static_memory_ignores_recent_rgb(gpu_models):
    model = make_estimator(CONFIG)
    t = model.context_size + CONFIG["history"]
    features = torch.randn(2, t, 384, 16, 16)
    geo = torch.rand(2, t, 4, 32, 32)
    geo[:, :, 3] = 1
    prop = torch.zeros(2, t, 18)
    camera = torch.eye(4).repeat(2, t, 1, 1)
    original = model(features, geo, prop, camera)
    shift = torch.tensor([0.2, -0.3, 0.4])
    camera[:, :, :3, 3] += shift
    translated = model(features, geo, prop, camera)
    torch.testing.assert_close(translated["hinge_origin"], original["hinge_origin"] + shift)
    torch.testing.assert_close(translated["contact_position"], original["contact_position"] + shift)
    features[:, model.context_size :] *= -100
    recent_changed = model(features, geo, prop, camera)
    for key in ("hinge_origin", "dimensions", "contact_local"):
        torch.testing.assert_close(recent_changed[key], translated[key])
    geo.requires_grad_(True)
    prediction = model(features, geo, prop, camera)
    (gradient,) = torch.autograd.grad(prediction["hinge_origin"].sum(), geo)
    assert gradient.isfinite().all()
    torch.testing.assert_close(
        gradient[:, : model.context_size, :3].sum((1, 3, 4)), torch.ones(2, 3)
    )
    with pytest.raises(ValueError, match="complete inspection"):
        model(features[:, -4:], geo[:, -4:], prop[:, -4:], camera[:, -4:])


def test_metric_pooling_excludes_far_depth_before_averaging(gpu_models):
    geometry = torch.ones(1, 4, 224, 224)
    geometry[:, :3, :112] = 20
    result = metric_geometry(geometry, CONFIG)
    assert not result[:, :, :16].any()
    torch.testing.assert_close(result[:, :, 16:], torch.ones_like(result[:, :, 16:]))


def test_live_inspection_memory_resets_and_unqualified_checkpoint_cannot_be_valid(gpu_models):
    class Backbone(torch.nn.Module):
        def forward(self, rgb):
            return torch.zeros(len(rgb), 384, 16, 16)

    config = dict(CONFIG, inspection=dict(sample_times_s=[0.1, 0.3]))
    model = make_estimator(config)
    with torch.no_grad():
        model.output.bias[23] = 10
    runner = ObservedEstimator(Backbone(), model, config)
    observation = dict(
        rgb=torch.full((60, 100, 3), 128, dtype=torch.uint8),
        depth_m=torch.ones(60, 100, 1),
        valid_depth=torch.ones(60, 100, 1, dtype=torch.bool),
        joint_position=torch.zeros(9),
        joint_velocity=torch.zeros(9),
        intrinsics=torch.tensor([[100.0, 0, 50], [0, 100, 30], [0, 0, 1]]),
        camera_world=torch.eye(4),
    )
    for i in range(8):
        observation.update(time_s=i / 10, frame=i + 1)
        result = runner.update(observation)
        assert not result.valid
    assert result.reason == "unqualified_confidence"
    assert len(runner.context) == 2
    model.confidence_qualified = True
    observation.update(time_s=0.8, frame=9)
    assert runner.update(observation).valid
    observation.update(time_s=1.2, frame=13)
    assert runner.update(observation).reason == "missing_inspection"
    assert not runner.context
