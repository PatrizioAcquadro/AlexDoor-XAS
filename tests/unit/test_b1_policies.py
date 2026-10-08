"""B1 family wiring and artifact contracts; no model inference or training."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from alexdoor_xas.action.b1 import ACTION_DIMS, DISCRETE_COLUMNS
from alexdoor_xas.action.spaces import A4_OBJ_CENTRIC_CHUNK
from alexdoor_xas.dataset.b1 import export_dataset
from alexdoor_xas.policies.act.config import ActModelCfg
from alexdoor_xas.policies.b1 import (
    B1Policy,
    batch_factories,
    load_b1_data,
    load_policy_payload,
    save_policy,
)
from alexdoor_xas.policies.diffusion.config import DiffusionModelCfg
from alexdoor_xas.policies.diffusion.data import MinMaxNormalizer
from helpers import TEST_ROBOT_REF, make_b1_episode


@pytest.fixture
def b1_data_root(tmp_path, b1_binding):
    train = make_b1_episode(b1_binding)
    dev = make_b1_episode(b1_binding, episode_id="dev", asset_id="right", sign=-1)
    dev.observations[:] += 1000
    return export_dataset(
        [train, dev],
        tmp_path / "dataset",
        b1_binding,
        TEST_ROBOT_REF,
        {"left": "train", "right": "development"},
    )


@pytest.mark.parametrize("family", ["act", "diffusion"])
@pytest.mark.parametrize("space", ACTION_DIMS)
def test_eight_batch_paths_and_fixed_discrete_scaling(b1_data_root, family, space):
    data = load_b1_data(b1_data_root, space)
    assert data.obs_dim == 22 and data.action_dim == ACTION_DIMS[space]
    assert data.stats.train_episode_ids == ("train-1",)
    assert data.stats.obs.mean.max() < 1  # the +1000 development intervention cannot leak
    train, development = batch_factories(data, family, 4, 2, 17)
    batch = next(iter(train(0)))
    assert batch["actions"].shape == (2, 4, ACTION_DIMS[space])
    assert batch["obs"].shape == (2, 22)
    assert next(iter(development()))["obs"].shape == (2, 22)
    if space == A4_OBJ_CENTRIC_CHUNK:
        values = batch["actions"][~batch["is_pad"]][:, DISCRETE_COLUMNS]
        assert set(np.unique(values)) <= ({0, 1} if family == "act" else {-1, 1})
    rows = data.dataset.records[0].actions
    normalizer = (
        data.stats.action
        if family == "act"
        else MinMaxNormalizer.from_norm_stats(data.stats.action)
    )
    np.testing.assert_allclose(normalizer.denormalize(normalizer.normalize(rows)), rows, atol=1e-14)


@pytest.mark.parametrize("family", ["act", "diffusion"])
@pytest.mark.parametrize("space", ACTION_DIMS)
def test_eight_checkpoint_envelopes_without_model_execution(
    tmp_path, b1_data_root, b1_binding, family, space
):
    data = load_b1_data(b1_data_root, space)
    # A scalar serialization fixture, deliberately not a neural model/valid policy.
    cfg = ActModelCfg() if family == "act" else DiffusionModelCfg()
    model = SimpleNamespace(
        obs_dim=data.obs_dim,
        action_dim=data.action_dim,
        cfg=cfg,
        state_dict=lambda: {"fixture": torch.ones(1)},
    )
    path = save_policy(tmp_path / "policy.pt", family, model, data)
    loaded, contract = load_policy_payload(
        path, family, binding=b1_binding, runtime_asset=TEST_ROBOT_REF
    )
    assert loaded.action_dim == ACTION_DIMS[space]
    assert contract == data.dataset.contract
    payload = torch.load(path, weights_only=True)
    payload["format"] = "alexdoor_xas." + family + ".v3"
    torch.save(payload, path)
    with pytest.raises(ValueError, match="unsupported checkpoint"):
        load_policy_payload(path, family, binding=b1_binding, runtime_asset=TEST_ROBOT_REF)


def test_stale_statistics_and_perception_mismatch_are_rejected(tmp_path, b1_data_root, b1_binding):
    data = load_b1_data(b1_data_root, A4_OBJ_CENTRIC_CHUNK)
    data.stats.obs.mean[:] += 1
    with pytest.raises(ValueError, match="training identities"):
        batch_factories(data, "act", 4, 2, 0)
    altered = b1_binding.to_dict()
    altered["artifacts"]["geometry"] = "f" * 64
    with pytest.raises(ValueError, match="perception mismatch"):
        load_b1_data(
            b1_data_root, A4_OBJ_CENTRIC_CHUNK, binding=type(b1_binding).from_dict(altered)
        )


@pytest.mark.parametrize("family", ["act", "diffusion"])
def test_a4_sources_preserve_segments_and_forbid_ensembling(b1_data_root, family):
    data = load_b1_data(b1_data_root, A4_OBJ_CENTRIC_CHUNK)
    expected = data.dataset.records[0].actions[:3]
    numerical = SimpleNamespace(stats=data.stats, chunk_size=3, predict=lambda obs: expected.copy())
    policy = B1Policy(numerical, family, data.dataset.contract, robot_asset=data.robot_asset)
    source = policy.chunk_source(lambda ctx: np.zeros(22))
    np.testing.assert_array_equal(source(None), expected)
    with pytest.raises(ValueError, match="Temporal ensembling"):
        policy.chunk_source(lambda ctx: np.zeros(22), temporal_ensemble=True)


@pytest.mark.parametrize("defect", ["keys", "dimension", "weights", "identity", "format"])
def test_checkpoint_rejects_corrupted_b1_payload(tmp_path, b1_data_root, b1_binding, defect):
    data = load_b1_data(b1_data_root, next(iter(ACTION_DIMS)))
    model = SimpleNamespace(
        obs_dim=data.obs_dim,
        action_dim=data.action_dim,
        cfg=ActModelCfg(),
        state_dict=lambda: {"fixture": torch.ones(1)},
    )
    path = save_policy(tmp_path / "policy.pt", "act", model, data)
    payload = torch.load(path, weights_only=True)
    if defect == "keys":
        payload["dataset"]["obs_keys"] = tuple(reversed(payload["dataset"]["obs_keys"]))
    elif defect == "dimension":
        payload["obs_dim"] += 1
    elif defect == "weights":
        payload["state_dict"]["fixture"][0] = float("nan")
    elif defect == "identity":
        payload["robot_asset"] = None
    else:
        payload["format"] = "alexdoor_xas.act.v3"
    torch.save(payload, path)
    with pytest.raises(ValueError):
        load_policy_payload(path, "act", binding=b1_binding, runtime_asset=TEST_ROBOT_REF)
