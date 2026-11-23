"""Essential synchronization, sensor-mask and completion regressions."""

import json

import numpy as np
import pytest

from alexdoor_xas.recording.b1 import B1Writer, robot_feedback_at, validate_episode
from helpers import raw_observation as obs


def test_causal_terminal_roundtrip_and_truth_separation(tmp_path):
    path = tmp_path / "episode.hdf5"
    meta = dict(asset_id="example", split="train", condition="nominal", control_dt=1 / 60)
    writer = B1Writer(path, meta, dict(depth_interval_m=[0.1, 5]))
    writer.observe(obs(), dict(angle=0.0))
    command = dict(
        time_s=0.0, joint_target=np.arange(7), tool_position=np.ones(3), torque_nm=np.ones(7)
    )
    writer.transition(command, obs(1 / 60, 2), dict(angle=0.1))
    writer.finish(
        dict(passed=True, released=True, hold_angle_deg=50.0, stop_reason="mechanical_stop")
    )
    writer.close()
    report = validate_episode(path)
    assert report["observations"] == 2
    assert report["duration_s"] == 1 / 60
    import h5py

    with h5py.File(path) as h5:
        assert set(h5) == {"observations", "commands", "annotations", "metadata", "robot_feedback"}
        assert "angle" not in h5["observations"]
        np.testing.assert_array_equal(h5["observations/depth_m"][0], obs()["depth_m"])
        np.testing.assert_array_equal(h5["commands/joint_target"][0], np.arange(7))
        assert robot_feedback_at(h5, 0).torque_nm is None
    with pytest.raises(FileExistsError):
        B1Writer(path, meta, {})
    with h5py.File(path, "r+") as h5:
        h5["metadata"].attrs["episode"] = json.dumps(dict(meta, split="test"))
    with pytest.raises(ValueError, match="Test recordings"):
        validate_episode(path)
    with h5py.File(path, "r+") as h5:
        h5["metadata"].attrs["episode"] = json.dumps(dict(meta, bounded_diagnostic=True))
    with pytest.raises(ValueError, match="outside the learning corpus"):
        validate_episode(path)
    assert validate_episode(path, diagnostic=True)["observations"] == 2


def test_invalid_masks_time_and_incomplete_recording(tmp_path):
    writer = B1Writer(tmp_path / "partial.hdf5", {}, dict(depth_interval_m=[0.1, 5]))
    with pytest.raises(ValueError, match="reset"):
        writer.observe(obs(1.0), {})
    observation = obs()
    observation["depth_m"][0, 0] = np.nan
    with pytest.raises(ValueError, match="mask"):
        writer.observe(observation, {})
    observation["valid_depth"][0, 0] = False
    writer.observe(observation, {})
    with pytest.raises(ValueError, match="timestamp"):
        writer.observe(observation, {})
    with pytest.raises(ValueError, match="Command"):
        writer.transition(dict(time_s=10.0), obs(1.0, 2), {})
    writer.close()
    with pytest.raises(ValueError, match="Incomplete"):
        validate_episode(tmp_path / "partial.hdf5")


def test_dark_material_is_not_an_empty_camera_frame(tmp_path):
    import h5py

    path = tmp_path / "dark.hdf5"
    writer = B1Writer(
        path,
        dict(asset_id="dark", split="train", condition="light", control_dt=0.1),
        dict(depth_interval_m=[0.1, 5]),
    )
    observation = obs()
    observation["rgb"][:] = 0
    observation["rgb"][0, 0] = 100
    writer.observe(observation, {})
    after = dict(observation, time_s=0.1, frame=2)
    writer.transition(dict(time_s=0.0, joint_target=np.zeros(7)), after, {})
    writer.finish(dict(passed=True, released=True, hold_angle_deg=50.0))
    writer.close()
    assert max(validate_episode(path)["rgb_mean"]) < 30
    with h5py.File(path, "r+") as h5:
        h5["observations/rgb"][0] = np.zeros((4, 6, 3), np.uint8)
    with pytest.raises(ValueError, match="Empty RGB"):
        validate_episode(path)


def test_campaign_membership_completeness_and_duplicate_rejection(tmp_path):
    import h5py

    from alexdoor_xas.recording.b1 import episode_paths

    entry = dict(asset_id="door", split="train", handedness="left", records_sha256="a" * 64)
    corpus = dict(doors=[entry])
    for condition in ("nominal", "light"):
        path = tmp_path / "door" / condition / "episode.hdf5"
        writer = B1Writer(
            path, dict(entry, condition=condition, control_dt=0.1), dict(depth_interval_m=[0.1, 5])
        )
        writer.observe(obs(), {})
        writer.transition(dict(time_s=0.0, joint_target=np.zeros(7)), obs(0.1, 2), {})
        writer.finish(dict(passed=True, released=True, hold_angle_deg=50.0))
        writer.close()
    paths = episode_paths(tmp_path, corpus)
    assert len(paths) == 2
    assert episode_paths(paths[0], corpus, complete_campaign=False) == [paths[0]]
    with h5py.File(paths[0], "r+") as h5:
        h5["metadata"].attrs["episode"] = json.dumps(
            dict(entry, condition="nominal", control_dt=0.1)
        )
    with pytest.raises(ValueError, match="directory"):
        episode_paths(paths[0], corpus, complete_campaign=False)
    with pytest.raises(ValueError, match="Duplicate"):
        episode_paths(tmp_path, corpus)
    paths[0].unlink()
    with pytest.raises(ValueError, match="Incomplete perception campaign"):
        episode_paths(tmp_path, corpus)
    assert len(episode_paths(tmp_path, corpus, complete_campaign=False)) == 1
    corpus["doors"][0]["records_sha256"] = "b" * 64
    with pytest.raises(ValueError, match="frozen corpus"):
        episode_paths(tmp_path, corpus, complete_campaign=False)
