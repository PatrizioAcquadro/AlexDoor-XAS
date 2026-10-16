"""Real encoder → raw replay → train-only batches → inference; no optimization."""

import json
from dataclasses import replace
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch

from alexdoor_xas.action.frames import ObjectFrame
from alexdoor_xas.action.spaces import A1_JOINT_DELTA, A2_EE_DELTA
from alexdoor_xas.assets.purdue import ARM_JOINTS, NECK_JOINTS
from alexdoor_xas.dataset.b1 import PILOT_PURPOSE, export_dataset, prepare_recording
from alexdoor_xas.dataset.normalize import compute_norm_stats
from alexdoor_xas.policies.act.config import ActModelCfg
from alexdoor_xas.policies.b1 import (
    B1Policy,
    batch_factories,
    load_b1_data,
    make_model,
    save_policy,
)
from alexdoor_xas.policies.diffusion.config import DiffusionModelCfg
from alexdoor_xas.policies.encoding import (
    FIXED_UNITS_RECIPE,
    IMAGE_DIM,
    LEARNED_RECIPE,
    RECIPE,
    RGBDEncoder,
    pilot_binding,
)
from alexdoor_xas.policies.observations import B1Observer
from alexdoor_xas.recording.b1 import PHASES, B1Writer, robot_feedback_at
from helpers import TEST_ROBOT_REF

pytestmark = pytest.mark.usefixtures("gpu_models")


@pytest.mark.parametrize("recipe", [RECIPE, FIXED_UNITS_RECIPE, LEARNED_RECIPE])
def test_causal_raw_roundtrip_and_models(tmp_path, recipe):
    inspection = json.loads(Path("configs/perception_inspection.json").read_text())
    binding = pilot_binding(inspection, recipe=recipe)
    live = B1Observer(None, binding, encoder=RGBDEncoder(binding))
    path = tmp_path / "raw.hdf5"
    metadata = dict(
        episode_id="pilot",
        asset_id="train-door",
        split="train",
        condition="nominal",
        purpose=PILOT_PURPOSE,
        control_dt=1 / 60,
        inspection=inspection,
    )
    calibration = dict(joint_names=list(ARM_JOINTS + NECK_JOINTS), depth_interval_m=[0.1, 5])
    writer = B1Writer(path, metadata, calibration)
    features = []
    tool = ObjectFrame(np.zeros(3), np.eye(3))
    for tick in range(25):
        depth = np.ones((32, 48, 1), dtype=np.float32)
        depth[0] = np.nan
        sample = dict(
            time_s=tick / 60,
            frame=tick,
            rgb=np.full((32, 48, 3), tick + 1, np.uint8),
            depth_m=depth,
            valid_depth=np.isfinite(depth),
            joint_position=np.full(9, tick * 0.001),
            joint_velocity=np.zeros(9),
            camera_world=np.eye(4),
            intrinsics=np.eye(3),
        )
        features.append(live.update(sample).features)
        labels = dict(phase=np.int8(PHASES.index("approach")))
        if tick == 0:
            writer.observe(sample, labels)
        else:
            writer.transition(
                dict(
                    time_s=(tick - 1) / 60,
                    joint_target=np.full(7, tick * 0.001),
                    tool_position=tool.origin,
                    tool_rotation=tool.rot,
                    phase=0,
                ),
                sample,
                labels,
            )
    writer.finish(dict(passed=True, termination="bounded_diagnostic"))
    writer.close()
    offline = B1Observer(None, binding, encoder=RGBDEncoder(binding))
    with pytest.raises(ValueError, match="split"):
        prepare_recording(path, offline, lambda q, c: tool, {"train-door": "development"})
    episode = prepare_recording(path, offline, lambda q, c: tool, {"train-door": "train"})
    np.testing.assert_array_equal(episode.observations, features)
    assert episode.observations[
        -1, IMAGE_DIM if recipe == LEARNED_RECIPE else 320
    ] == pytest.approx(13 / 255)
    live.reset()
    assert live.update(sample | dict(time_s=0, frame=0)).feature_support.recent.acquired_s == 0
    dev = replace(
        episode,
        episode_id="dev",
        asset_id="dev-door",
        split="development",
        observations=episode.observations + 1000,
        source_recording=None,
    )
    splits = {"train-door": "train", "dev-door": "development"}
    with pytest.raises(ValueError, match="physical episode"):
        export_dataset(
            [replace(dev, source_recording=str(path))],
            tmp_path / "wrong",
            binding,
            TEST_ROBOT_REF,
            splits,
        )
    root = export_dataset([episode, dev], tmp_path / "data", binding, TEST_ROBOT_REF, splits)
    with h5py.File(root / "episode_000000.hdf5") as source:
        np.testing.assert_array_equal(source["raw/observations/rgb"][-1], sample["rgb"])
        assert len(source["raw/commands/phase"]) == 24
    with h5py.File(path) as source:
        assert robot_feedback_at(source, 0).torque_nm is None
    for space in (A1_JOINT_DELTA, A2_EE_DELTA):
        data = load_b1_data(root, space)
        assert data.stats.obs.max.max() < 1000
        with pytest.raises(ValueError, match="train-only"):
            compute_norm_stats(data.dataset, ["dev"], data.stats.obs_keys)
        for family in ("act", "diffusion"):
            cfg = ActModelCfg(chunk_size=12) if family == "act" else DiffusionModelCfg(horizon=12)
            model = make_model(data, family, cfg)
            batches, _ = batch_factories(data, family, 12, 2, 0)
            batch = next(iter(batches(0)))
            assert np.isfinite(batch["obs"]).all()
            if recipe == LEARNED_RECIPE:
                model.train()
                obs_tensor = torch.as_tensor(batch["obs"], dtype=torch.float32, device="cuda")
                projection = model.obs_proj if family == "act" else model.obs_mlp[0]
                projection(obs_tensor).square().mean().backward()
                grad = projection.images[0].weight.grad
                assert grad is not None and torch.isfinite(grad).all() and grad.abs().sum() > 0
                model.zero_grad(set_to_none=True)
                model.eval()
                modified = obs_tensor.clone()
                modified[:, :IMAGE_DIM] = 0
                assert not torch.equal(projection(obs_tensor), projection(modified))
            saved = save_policy(tmp_path / f"{family}-{space}.pt", family, model, data)
            policy = B1Policy.from_checkpoint(
                saved, family, binding=binding, runtime_asset=TEST_ROBOT_REF, num_inference_steps=10
            )
            incompatible = pilot_binding(
                inspection, recipe=FIXED_UNITS_RECIPE if recipe == RECIPE else RECIPE
            )
            with pytest.raises(ValueError, match="mismatch"):
                B1Policy.from_checkpoint(
                    saved, family, binding=incompatible, runtime_asset=TEST_ROBOT_REF
                )
            assert next(policy.policy.model.parameters()).is_cuda
            policy.reset(23)
            expected = policy.predict(features[-1])
            second = B1Policy.from_checkpoint(
                saved, family, binding=binding, runtime_asset=TEST_ROBOT_REF, num_inference_steps=10
            )
            second.reset(23)
            np.testing.assert_array_equal(expected, second.predict(features[-1]))
            assert expected.shape == (12, data.action_dim)
            assert policy.chunk_source(lambda x: x, n_action_steps=6)(features[-1]).shape[0] == 6
            torch.cuda.synchronize()


def test_learned_missing_depth_freeze_and_generation():
    inspection = json.loads(Path("configs/perception_inspection.json").read_text())
    binding = pilot_binding(inspection, recipe=LEARNED_RECIPE, depth_interval_m=(0.01, 100000))
    encoder = RGBDEncoder(binding)
    sample = dict(
        rgb=np.full((60, 96, 3), 128, np.uint8),
        depth_m=np.full((60, 96, 1), np.nan, np.float32),
        valid_depth=np.zeros((60, 96, 1), bool),
        time_s=0,
        frame=0,
    )
    empty = encoder.encode(sample).reshape(5, 60, 96)
    assert np.isfinite(empty).all() and np.all(empty[3:] == 0)
    sample.update(
        depth_m=np.full((60, 96, 1), 6, np.float32), valid_depth=np.ones((60, 96, 1), bool)
    )
    full = encoder.encode(sample).reshape(5, 60, 96)
    np.testing.assert_allclose(full[3:], 1, atol=1e-6)
    frozen = encoder.update(sample | dict(time_s=encoder.static_until, frame=1), generation=0)
    original = frozen.vector.copy()
    current = encoder.update(
        sample
        | dict(time_s=encoder.static_until + 0.1, frame=2, rgb=np.zeros((60, 96, 3), np.uint8)),
        generation=0,
    )
    np.testing.assert_array_equal(current.static.values, frozen.static.values)
    np.testing.assert_array_equal(frozen.vector, original)
    assert current.recent.acquired_s > frozen.recent.acquired_s
    restarted = encoder.update(
        sample | dict(time_s=0, frame=0, rgb=np.zeros((60, 96, 3), np.uint8)), generation=1
    )
    assert restarted.static.generation == restarted.recent.generation == 1
    assert restarted.static.acquired_s == restarted.recent.acquired_s == 0
    assert np.all(restarted.static.values.reshape(5, 60, 96)[:3] == 0)
