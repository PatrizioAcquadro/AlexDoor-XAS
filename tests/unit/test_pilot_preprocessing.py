"""Fixed observation units preserve sensor validity and legacy recipe behavior."""

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from test_b1_pilot import pilot_episode

from alexdoor_xas.action.spaces import A2_EE_DELTA
from alexdoor_xas.dataset.b1 import export_dataset
from alexdoor_xas.dataset.normalize import NormStats
from alexdoor_xas.policies.b1 import batch_factories, check_stats, load_b1_data
from alexdoor_xas.policies.encoding import (
    FIXED_UNITS_RECIPE,
    RGBDEncoder,
    pilot_binding,
)
from helpers import TEST_ROBOT_REF


def test_units_conversion_alone_does_not_repair_zscore():
    train = np.array([[1.000, 1.000], [1.001, 1.002]])
    dev = np.array([[1.5, 1.6]])
    original = NormStats.from_rows([train])
    converted = NormStats.from_rows([train / 5])
    np.testing.assert_allclose(original.normalize(dev), converted.normalize(dev / 5))
    assert (np.abs(converted.normalize(dev / 5)) > 10).all()


def test_policy_depth_scale_is_separate_from_sensor_interval():
    inspection = json.loads(Path("configs/perception_inspection.json").read_text())
    args = dict(depth_interval_m=(0.01, 100000))
    legacy = RGBDEncoder(pilot_binding(inspection, **args), device="cpu")
    fixed = RGBDEncoder(pilot_binding(inspection, recipe=FIXED_UNITS_RECIPE, **args), device="cpu")
    sample = dict(
        rgb=np.full((8, 8, 3), 128, np.uint8),
        depth_m=np.full((8, 8, 1), 6, np.float32),
        valid_depth=np.ones((8, 8, 1), bool),
    )
    sample["depth_m"][0, 0] = np.nan
    sample["valid_depth"][0, 0] = False
    old, new = legacy.encode(sample).reshape(5, 64), fixed.encode(sample).reshape(5, 64)
    np.testing.assert_array_equal(new[[0, 1, 2, 4]], old[[0, 1, 2, 4]])
    np.testing.assert_allclose(new[3, 1:], 1.2)
    assert new[3, 0] == new[4, 0] == 0
    assert new[4, 1] == 1  # 6m remains valid although the policy scale is 5m.
    with pytest.raises(ValueError, match="sensor interval"):
        fixed.encode(sample | dict(valid_depth=np.ones((8, 8, 1), bool)))


def test_fixed_units_train_only_actions_and_identical_training_transform(tmp_path, b1_binding):
    from alexdoor_xas.policies.common.b1_contract import PILOT_SCHEMA, PilotBinding

    value = dict(
        schema=PILOT_SCHEMA,
        offline_passed=False,
        dynamic_passed=False,
        frozen=False,
        config=b1_binding.config
        | dict(encoder=FIXED_UNITS_RECIPE, geometry_profile="operational-v1", warmup_s=0),
    )
    value["config"]["inspection"]["sample_times_s"] = [1 / 60]
    value["config"]["inspection"]["waypoints"] = [[0, 0, 0], [1 / 60, 0, 0]]
    binding = PilotBinding.from_dict(value)
    train = pilot_episode(binding)
    dev = replace(
        train,
        episode_id="dev",
        asset_id="dev",
        split="development",
        observations=train.observations + 999,
    )
    root = export_dataset(
        [train, dev],
        tmp_path / "data",
        binding,
        TEST_ROBOT_REF,
        {"left": "train", "dev": "development"},
    )
    data = load_b1_data(root, A2_EE_DELTA)
    np.testing.assert_array_equal(data.stats.obs.mean, 0)
    np.testing.assert_array_equal(data.stats.obs.std, 1)
    assert data.stats.obs.max.max() < 999
    assert data.stats.train_episode_ids == ("pilot",)
    for family in ("act", "diffusion"):
        factory, _ = batch_factories(data, family, 2, 1, 17)
        batch = next(iter(factory(0)))
        expected = np.clip(train.observations[0], -10, 10)
        np.testing.assert_array_equal(batch["obs"][0], expected)
    data.stats.obs.std[0] = 2
    with pytest.raises(ValueError, match="Fixed-unit"):
        check_stats(data.stats, data.dataset.contract)


def test_fixed_units_preserve_saved_pooling_roundoff():
    inspection = json.loads(Path("configs/perception_inspection.json").read_text())
    options = dict(depth_interval_m=(0.01, 100000))
    legacy = RGBDEncoder(pilot_binding(inspection, **options), device="cpu")
    fixed = RGBDEncoder(
        pilot_binding(inspection, recipe=FIXED_UNITS_RECIPE, **options), device="cpu"
    )
    depth = np.linspace(0.1, 40, 32 * 48, dtype=np.float32).reshape(32, 48, 1)
    depth[0, 0] = np.nan
    sample = dict(
        rgb=np.full((32, 48, 3), 128, np.uint8),
        depth_m=depth,
        valid_depth=np.isfinite(depth),
    )
    saved = legacy.encode(sample).astype(np.float64)
    saved[192:256] *= 100000 / 5
    np.testing.assert_array_equal(fixed.encode(sample), saved.astype(np.float32))
