"""Fixed observation units preserve sensor validity and legacy recipe behavior."""

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from alexdoor_xas.action.spaces import A2_EE_DELTA
from alexdoor_xas.dataset.b1 import export_dataset
from alexdoor_xas.policies.b1 import batch_factories, check_stats, load_b1_data
from alexdoor_xas.policies.encoding import (
    FIXED_UNITS_RECIPE,
    RGBDEncoder,
    pilot_binding,
)
from helpers import TEST_ROBOT_REF, pilot_episode


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


@pytest.mark.parametrize(
    "declared,requested,expected",
    [
        ("rgbd-conv-v1", None, "rgbd-conv-v1"),
        (None, "rgbd-grid-v1", "rgbd-grid-v1"),
        (None, "rgbd-grid-fixed-units-v2", "rgbd-grid-fixed-units-v2"),
    ],
)
def test_explicit_protocol_recipe_preserves_historical_identity(declared, requested, expected):
    from alexdoor_xas.policies.encoding import protocol_recipe

    protocol = {} if declared is None else {"recipe": declared}
    before = protocol.copy()
    assert protocol_recipe(protocol, requested) == expected
    assert protocol == before


@pytest.mark.parametrize(
    "protocol,requested",
    [
        ({}, None),
        ({"recipe": "unknown"}, None),
        ({"recipe": "rgbd-conv-v1"}, "rgbd-grid-v1"),
    ],
)
def test_missing_or_conflicting_encoder_recipe_is_refused(protocol, requested):
    from alexdoor_xas.policies.encoding import protocol_recipe

    with pytest.raises(ValueError, match="recipe|encoder"):
        protocol_recipe(protocol, requested)


@pytest.mark.parametrize(
    "script,argv",
    [
        ("pilot_offline", ["prepare"]),
        ("pilot_physics", ["collect", "--asset-root", "/unused", "--asset-id", "unused"]),
    ],
)
def test_pilot_commands_require_protocol_before_starting_work(tmp_path, script, argv):
    import runpy

    main = runpy.run_path(str(Path("scripts") / f"{script}.py"))["main"]
    extra = ["--root", str(tmp_path)] if script == "pilot_offline" else []
    with pytest.raises(SystemExit) as stopped:
        main(
            [*argv, *extra, "--output", str(tmp_path / "result")]
            if script == "pilot_physics"
            else [*argv, *extra]
        )
    assert stopped.value.code == 2
    assert not (tmp_path / "result").exists()


def test_offline_refuses_dataset_recipe_mismatch_before_cuda(tmp_path):
    import runpy

    protocol = tmp_path / "protocol.json"
    protocol.write_text(json.dumps({"recipe": "rgbd-conv-v1"}))
    (tmp_path / "data").mkdir()
    (tmp_path / "data/meta.json").write_text(
        json.dumps({"perception": {"config": {"encoder": "rgbd-grid-v1"}}})
    )
    main = runpy.run_path("scripts/pilot_offline.py")["main"]
    with pytest.raises(SystemExit) as stopped:
        main(
            [
                "evaluate",
                "--root",
                str(tmp_path),
                "--protocol",
                str(protocol),
                "--checkpoints-root",
                str(tmp_path / "checkpoints"),
            ]
        )
    assert stopped.value.code == 2
    assert not (tmp_path / "evaluation").exists()


def test_new_preparation_refuses_historical_grid_without_changing_protocol(tmp_path):
    import runpy

    protocol = tmp_path / "protocol.json"
    contents = json.dumps({"recipe": "rgbd-grid-v1"})
    protocol.write_text(contents)
    main = runpy.run_path("scripts/pilot_offline.py")["main"]
    with pytest.raises(SystemExit) as stopped:
        main(["prepare", "--root", str(tmp_path), "--protocol", str(protocol)])
    assert stopped.value.code == 2 and protocol.read_text() == contents
    assert not (tmp_path / "data").exists()
