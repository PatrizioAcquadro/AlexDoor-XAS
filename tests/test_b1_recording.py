"""Essential synchronization, sensor-mask and completion regressions."""

import json

import numpy as np
import pytest

from alexdoor_xas.recording.b1 import B1Writer, validate_episode


def obs(t=0, frame=1):
    return dict(
        time_s=t,
        frame=frame,
        rgb=np.full((4, 6, 3), 100, np.uint8),
        depth_m=np.ones((4, 6, 1), np.float32),
        valid_depth=np.ones((4, 6, 1), bool),
        joint_position=np.zeros(9),
        joint_velocity=np.zeros(9),
        camera_world=np.eye(4),
        intrinsics=np.eye(3),
    )


def test_causal_terminal_roundtrip_and_truth_separation(tmp_path):
    path = tmp_path / "episode.hdf5"
    meta = dict(asset_id="example", split="train", condition="nominal", control_dt=1 / 60)
    writer = B1Writer(path, meta, dict(depth_interval_m=[0.1, 5]))
    writer.observe(obs(), dict(angle=0.0))
    command = dict(time_s=0.0, joint_target=np.arange(7), tool_position=np.ones(3))
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
        assert set(h5) == {"observations", "commands", "annotations", "metadata"}
        assert "angle" not in h5["observations"]
        np.testing.assert_array_equal(h5["commands/joint_target"][0], np.arange(7))
    with pytest.raises(FileExistsError):
        B1Writer(path, meta, {})
    with h5py.File(path, "r+") as h5:
        h5["metadata"].attrs["episode"] = json.dumps(dict(meta, split="test"))
    with pytest.raises(ValueError, match="Test recordings"):
        validate_episode(path)


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
