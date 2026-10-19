"""Actual checkpoint -> ingress -> A1 runner on CUDA; no simulator or updates."""

from dataclasses import replace

import numpy as np
import pytest
import torch

from alexdoor_xas.dataset.normalize import DatasetNormStats, NormStats
from alexdoor_xas.policies.act.execution import temporal_a1_runner
from alexdoor_xas.policies.act.temporal import OBS_KEYS, save_temporal_policy, temporal_model
from helpers import TEST_ROBOT_REF
from unit.test_temporal_execution import replay_runner  # noqa: F401


def test_checkpoint_runner_order_replanning_compatibility_and_reset(tmp_path, request):
    assert torch.cuda.is_available(), "Actual CUDA required"
    runner, io, _, execution, _, _ = request.getfixturevalue("replay_runner")
    cfg = execution.observations.config
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

    def identity(width):
        return NormStats(np.zeros(width), np.ones(width), np.zeros(width), np.ones(width), 1)

    stats = DatasetNormStats(identity(7), identity(32), OBS_KEYS, ("fixture",), "A1_joint_delta")
    checkpoint = save_temporal_policy(tmp_path / "temporal.pt", model, stats, TEST_ROBOT_REF)
    kwargs = dict(
        config=cfg,
        episode_id="saved-events",
        events=runner.observer.events,
        admission_source=runner.admission_source,
    )
    maintained = temporal_a1_runner(checkpoint, runner.observer.observer, io, **kwargs)
    predictions = []
    predict = maintained.policy.policy.predict

    def record(snapshot, **options):
        chunk = predict(snapshot, **options)
        predictions.append(chunk.copy())
        return chunk

    maintained.policy.policy.predict = record
    result = maintained.run(max_ticks=13)
    assert result.reason == "time_budget" and not result.predicted_complete
    assert len(predictions) == 3 and len(io.commands) == 13
    np.testing.assert_array_equal(
        [c.value for c in io.commands], np.concatenate([p[:6] for p in predictions])[:13]
    )
    assert maintained.policy.policy.device.type == "cuda"
    assert not maintained.policy.policy.model.obs_proj.cache
    assert maintained.run(max_ticks=1).control_steps == 1
    np.testing.assert_allclose(io.commands[0].value, predictions[0][0], atol=1e-6)
    io.robot_asset = replace(TEST_ROBOT_REF, sha256="e" * 64)
    with pytest.raises(ValueError, match="robot asset"):
        temporal_a1_runner(checkpoint, runner.observer.observer, io, **kwargs)
    io.robot_asset = TEST_ROBOT_REF
    io.dt = 1 / 30
    with pytest.raises(ValueError, match="control interval"):
        temporal_a1_runner(checkpoint, runner.observer.observer, io, **kwargs)
    io.dt = 1 / 60
    loaded = torch.load(checkpoint, weights_only=False)
    loaded["meta"]["temporal_contract"]["arm_joints"].reverse()
    torch.save(loaded, tmp_path / "wrong-joints.pt")
    with pytest.raises(ValueError, match="robot/action contract"):
        temporal_a1_runner(tmp_path / "wrong-joints.pt", runner.observer.observer, io, **kwargs)
