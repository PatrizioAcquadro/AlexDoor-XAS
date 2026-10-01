"""Matched B1 physical identities, terminal observations and segment sampling."""

from dataclasses import replace

import numpy as np
import pytest

from alexdoor_xas.action.b1 import ACTION_DIMS
from alexdoor_xas.action.spaces import A4_OBJ_CENTRIC_CHUNK
from alexdoor_xas.dataset.b1 import B1Dataset, export_dataset, validate_episode
from alexdoor_xas.dataset.sampling import ChunkSampler
from alexdoor_xas.policies.common.b1_contract import OBS_KEYS, validate_contract
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


def test_contract_rejects_legacy_or_reordered_joints(b1_binding):
    from alexdoor_xas.policies.common.b1_contract import policy_contract

    with pytest.raises(ValueError, match="legacy"):
        validate_contract({"obs_dim": b1_binding.obs_dim})
    contract = policy_contract(b1_binding, next(iter(ACTION_DIMS)))
    contract["arm_joints"].reverse()
    with pytest.raises(ValueError, match="mismatch"):
        validate_contract(contract)


def test_compilation_cannot_silently_mix_operational_observations_with_legacy_release(b1_binding):
    from alexdoor_xas.dataset.b1 import compile_episode
    from test_b1_rollout import operational_observation
    from test_operational_admission import operational_estimate

    observations = [replace(operational_observation(operational_estimate(time=t), t), frame=i)
                    for i, t in enumerate((0, 0.1))]
    tool = observations[0].estimate.operational.contact.world_pose
    with pytest.raises(ValueError, match="profile differs"):
        compile_episode(episode_id="numeric", asset_id="left", asset_splits=SPLITS,
                        observations=observations, tools=[tool, tool], joint_targets=[np.zeros(7)],
                        goals=[tool], stages=["approach"], binding=b1_binding)


def test_raw_preparation_never_promotes_engineering_recordings(tmp_path):
    from alexdoor_xas.dataset.b1 import prepare_recording
    from alexdoor_xas.recording.b1 import B1Writer
    from test_b1_recording import obs

    path = tmp_path / "episode.hdf5"
    writer = B1Writer(
        path,
        dict(
            asset_id="left",
            split="train",
            condition="nominal",
            control_dt=1 / 60,
            purpose="perception_engineering_not_matched_policy_dataset",
        ),
        dict(depth_interval_m=[0.1, 5]),
    )
    writer.observe(obs(), dict(angle=0.0))
    writer.transition(dict(time_s=0.0, joint_target=np.zeros(7)), obs(1 / 60, 2), dict(angle=0.1))
    writer.finish(dict(passed=True, released=True, hold_angle_deg=50.0))
    writer.close()
    with pytest.raises(ValueError, match="engineering recordings"):
        prepare_recording(path, None, None, {"left": "train"})


def test_raw_preparation_uses_provider_without_model_or_device_internals(tmp_path, b1_binding):
    from alexdoor_xas.action.b1 import STAGES
    from alexdoor_xas.action.frames import ObjectFrame
    from alexdoor_xas.assets.purdue import ARM_JOINTS, NECK_JOINTS
    from alexdoor_xas.dataset.b1 import PURPOSE, prepare_recording
    from alexdoor_xas.policies.observations import B1Observer
    from alexdoor_xas.recording.b1 import PHASES, B1Writer
    from test_b1_observations import fake_estimator
    from test_b1_recording import obs

    path = tmp_path / "episode.hdf5"
    writer = B1Writer(
        path,
        dict(
            episode_id="matched",
            asset_id="left",
            split="train",
            condition="nominal",
            control_dt=0.1,
            purpose=PURPOSE,
            inspection=b1_binding.config["inspection"],
        ),
        dict(depth_interval_m=[0.1, 5], joint_names=list(ARM_JOINTS + NECK_JOINTS)),
    )
    writer.observe(obs(), {})
    pose = ObjectFrame(np.array([0.04, 0.3, 0.5]), np.eye(3))
    for i, stage in enumerate(s for s in STAGES for _ in range(2)):
        writer.transition(
            dict(
                time_s=i * 0.1,
                joint_target=np.zeros(7),
                tool_position=pose.origin,
                tool_rotation=pose.rot,
                phase=PHASES.index(stage),
            ),
            obs((i + 1) * 0.1, i + 2),
            {},
        )
    writer.finish(dict(passed=True, released=True, hold_angle_deg=50.0))
    writer.close()
    provider = fake_estimator(b1_binding)
    observer = B1Observer(provider, b1_binding)
    episode = prepare_recording(path, observer, lambda joints, calibration: pose, SPLITS)
    assert episode.observations.shape == (11, b1_binding.obs_dim)
    assert len(episode.actions[A4_OBJ_CENTRIC_CHUNK]) == 5
    assert isinstance(provider.calls[-1]["rgb"], np.ndarray)
    assert not hasattr(provider, "estimator") and not hasattr(provider, "device")
