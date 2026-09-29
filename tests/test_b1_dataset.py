"""Matched B1 physical identities, terminal observations and segment sampling."""

from dataclasses import replace

import numpy as np
import pytest

from alexdoor_xas.action.b1 import ACTION_DIMS
from alexdoor_xas.action.spaces import A4_OBJ_CENTRIC_CHUNK
from alexdoor_xas.dataset.b1 import B1Dataset, export_dataset, validate_episode
from alexdoor_xas.dataset.sampling import ChunkSampler
from alexdoor_xas.policies.common.b1_contract import OBS_KEYS, PerceptionBinding, validate_contract
from conftest import TEST_ROBOT_REF, make_b1_episode

SPLITS = {"left": "train", "right": "development"}


def test_all_views_share_episode_observations_and_terminal_state(tmp_path, b1_binding):
    train = make_b1_episode(b1_binding)
    dev = make_b1_episode(b1_binding, episode_id="dev-1", asset_id="right", sign=-1)
    root = export_dataset([train, dev], tmp_path / "b1", b1_binding, TEST_ROBOT_REF, SPLITS)
    for space in ACTION_DIMS:
        view = B1Dataset(root, space, binding=b1_binding)
        validate_contract(view.contract, b1_binding, space)
        assert view.episode_ids == ["train-1", "dev-1"]
        record = view.by_id("train-1")
        np.testing.assert_array_equal(record.terminal_observation, train.observations[-1])
        sampler = ChunkSampler(view, 8, OBS_KEYS, ["train-1"])
        last = sampler.sample(len(sampler) - 1)
        assert last.actions.shape == (8, ACTION_DIMS[space])
        assert last.is_pad.tolist() == [False] + [True] * 7
        np.testing.assert_array_equal(last.obs, train.observations[record.starts[-1]])
        if space == A4_OBJ_CENTRIC_CHUNK:
            assert len(sampler) == len(train.actions[space])
    with pytest.raises(FileExistsError):
        export_dataset([train], root, b1_binding, TEST_ROBOT_REF, SPLITS)


@pytest.mark.parametrize("defect", ["split", "terminal", "gap", "early_end", "missing_path"])
def test_dataset_rejects_unmatched_or_incomplete_contract(b1_binding, defect):
    episode = make_b1_episode(b1_binding)
    if defect == "split":
        episode = replace(episode, split="test")
    elif defect == "terminal":
        episode = replace(episode, observations=episode.observations[:-1])
    elif defect == "gap":
        episode.starts[A4_OBJ_CENTRIC_CHUNK][1] += 1
    elif defect == "early_end":
        episode.actions[A4_OBJ_CENTRIC_CHUNK][0, 16] = 1
    else:
        episode.actions.pop(next(iter(episode.actions)))
    with pytest.raises(ValueError):
        validate_episode(episode, b1_binding, SPLITS)


@pytest.mark.parametrize("gate", ["offline_passed", "dynamic_passed", "frozen"])
def test_unqualified_or_unfrozen_release_cannot_bind(b1_binding, gate):
    release = b1_binding.to_dict()
    release[gate] = False
    with pytest.raises(ValueError, match="qualified and frozen"):
        PerceptionBinding.from_dict(release)


def test_contract_rejects_legacy_or_reordered_joints(b1_binding):
    from alexdoor_xas.policies.common.b1_contract import policy_contract

    with pytest.raises(ValueError, match="legacy"):
        validate_contract({"obs_dim": b1_binding.obs_dim})
    contract = policy_contract(b1_binding, next(iter(ACTION_DIMS)))
    contract["arm_joints"].reverse()
    with pytest.raises(ValueError, match="mismatch"):
        validate_contract(contract)
