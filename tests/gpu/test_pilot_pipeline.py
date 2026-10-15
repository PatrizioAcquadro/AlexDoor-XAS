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
from alexdoor_xas.policies.encoding import RGBDEncoder, pilot_binding
from alexdoor_xas.policies.observations import B1Observer
from alexdoor_xas.recording.b1 import PHASES, B1Writer, robot_feedback_at
from helpers import TEST_ROBOT_REF

pytestmark = pytest.mark.usefixtures("gpu_models")


def test_causal_raw_roundtrip_and_models(tmp_path):
    inspection = json.loads(Path("configs/perception_inspection.json").read_text())
    binding = pilot_binding(inspection)
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
    assert episode.observations[-1, 320] == pytest.approx(13 / 255)
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
            saved = save_policy(tmp_path / f"{family}-{space}.pt", family, model, data)
            policy = B1Policy.from_checkpoint(
                saved, family, binding=binding, runtime_asset=TEST_ROBOT_REF, num_inference_steps=10
            )
            assert next(policy.policy.model.parameters()).is_cuda
            assert policy.predict(features[-1]).shape == (12, data.action_dim)
            assert policy.chunk_source(lambda x: x, n_action_steps=6)(features[-1]).shape[0] == 6
            torch.cuda.synchronize()
