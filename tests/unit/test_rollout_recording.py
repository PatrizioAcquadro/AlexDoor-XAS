"""Consumed input joins, failed outcomes and lossless separate-clock retention."""

import json
from types import SimpleNamespace

import h5py
import numpy as np
import pytest

from alexdoor_xas.policies.observations import FeatureBlock, VisualEncoding
from alexdoor_xas.recording.rollout import RolloutWriter


def sensor(tick):
    depth = np.ones((4, 6, 1), np.float32)
    # Preserve the actual missing-depth bit pattern, without normalizing NaNs.
    depth.view(np.uint32)[0, 0, 0] = 0x7FC00017
    return dict(
        time_s=tick / 60,
        frame=tick + 1,
        rgb=np.full((4, 6, 3), tick, np.uint8),
        depth_m=depth,
        valid_depth=np.isfinite(depth),
        joint_position=np.full(9, tick / 60),
        joint_velocity=np.zeros(9),
        camera_world=np.eye(4),
        intrinsics=np.eye(3),
    )


def observation(tick, recent_tick, *, generation=0, support=True):
    block = FeatureBlock(
        np.zeros(1), recent_tick / 60, recent_tick / 60, recent_tick + 1, generation
    )
    return SimpleNamespace(
        time_s=tick / 60,
        frame=tick + 1,
        generation=generation,
        feature_support=VisualEncoding(block, block) if support else None,
    )


def writer(path):
    return RolloutWriter(
        path,
        dict(purpose="learned_evaluation"),
        dict(pixel_center_offset_xy=[0.5, 0.5]),
        control_dt=1 / 60,
        image_period_s=0.1,
    )


def test_lossless_input_joins_and_complete_failure(tmp_path):
    path = tmp_path / "failure.hdf5"
    log = writer(path)
    for tick in range(9):
        # Tick 7 represents an extra frame actually consumed by a producer.
        recent = 7 if tick >= 7 else (6 if tick >= 6 else 0)
        log.observe(sensor(tick), observation(tick, recent), dict(actual_tool=np.zeros(3)))
        if tick:
            log.command(dict(time_s=(tick - 1) / 60, joint_target=np.full(7, tick)))
            log.feedback(
                [
                    dict(time_s=(2 * tick - 1) / 120, torque_nm=np.zeros(7)),
                    dict(time_s=2 * tick / 120, torque_nm=np.ones(7)),
                ],
                [
                    dict(time_s=(2 * tick - 1) / 120, force_n=0.0),
                    dict(time_s=2 * tick / 120, force_n=0.1),
                ],
            )
    log.finish(dict(passed=False, termination="contact_lost", learned_commands=8))
    log.close()
    with h5py.File(path) as h5:
        assert h5.attrs["complete"]
        assert not json.loads(h5["metadata"].attrs["outcome"])["passed"]
        assert len(h5["control/time_s"]) == 9
        assert len(h5["commands/time_s"]) == 8
        assert len(h5["physics_feedback/time_s"]) == len(h5["evaluator/time_s"]) == 16
        np.testing.assert_array_equal(h5["captures/frame"][:], [1, 7, 8])
        np.testing.assert_array_equal(h5["control/image_indices"][5], [0, 0, 0, 0])
        np.testing.assert_array_equal(h5["control/image_indices"][6], [1, 0, 0, 1])
        np.testing.assert_array_equal(h5["control/image_indices"][8], [2, 0, 1, 2])
        for row, tick in enumerate((0, 6, 7)):
            for key in ("rgb", "depth_m", "valid_depth"):
                np.testing.assert_array_equal(
                    h5[f"captures/{key}"][row].view(np.uint8), sensor(tick)[key].view(np.uint8)
                )
        assert "force_n" not in h5["captures"] and "force_n" not in h5["control"]
        assert h5["captures/depth_m"].compression is None
        assert h5["captures/depth_m"].chunks == (1, 4, 6, 1)
    with pytest.raises(FileExistsError):
        writer(path)


def test_missing_features_retained_and_interruption_stays_incomplete(tmp_path):
    path = tmp_path / "interrupted.hdf5"
    log = writer(path)
    log.observe(sensor(0), observation(0, 0, support=False), {})
    with pytest.raises(ValueError, match="control_clock"):
        log.observe(sensor(0), observation(0, 0), {})
    log.observe(sensor(1), observation(1, 0, support=False), {})
    with pytest.raises(ValueError, match="incomplete"):
        log.finish(dict(passed=False, termination="interrupted"))
    log.close()
    with h5py.File(path) as h5:
        assert not h5.attrs["complete"]
        np.testing.assert_array_equal(h5["control/image_indices"][:], np.full((2, 4), -1))
        assert len(h5["captures/frame"]) == 1


def test_future_features_and_command_or_feedback_clock_refuse(tmp_path):
    log = writer(tmp_path / "bad-clock.hdf5")
    log.observe(sensor(0), observation(0, 0), {})
    future = observation(1, 1)
    future.feature_support = VisualEncoding(
        future.feature_support.static,
        FeatureBlock(np.zeros(1), 2 / 60, 2 / 60, 2, 0),
    )
    with pytest.raises(ValueError, match="feature_clock"):
        log.observe(sensor(1), future, {})
    log.observe(sensor(1), observation(1, 0), {})
    with pytest.raises(ValueError, match="noncausal_recorded_command"):
        log.command(dict(time_s=1 / 60, joint_target=np.zeros(7)))
    log.command(dict(time_s=0.0, joint_target=np.zeros(7)))
    with pytest.raises(ValueError, match="physics_clock"):
        log.feedback([dict(time_s=1.0, torque_nm=np.zeros(7))], [])
    log.feedback([dict(time_s=1 / 60, torque_nm=np.zeros(7))], [])
    with pytest.raises(ValueError, match="physics_clock"):
        log.feedback([dict(time_s=1 / 60, torque_nm=np.zeros(7))], [])
    log.close()


def test_unrecorded_static_frame_cannot_be_invented(tmp_path):
    log = writer(tmp_path / "missing-frame.hdf5")
    log.observe(sensor(0), observation(0, 0), {})
    obs = observation(1, 0)
    obs.feature_support = VisualEncoding(
        FeatureBlock(np.zeros(1), 1 / 60, 1 / 60, 99, 0), obs.feature_support.recent
    )
    with pytest.raises(ValueError, match="unrecorded_policy_image"):
        log.observe(sensor(1), obs, {})
    log.close()


def test_chunk_boundaries_partial_flush_and_invalid_row(tmp_path):
    path = tmp_path / "chunks.hdf5"
    log = writer(path)
    for tick in range(133):
        log.observe(sensor(tick), observation(tick, tick - tick % 6), {})
        if tick:
            log.command(dict(time_s=(tick - 1) / 60, joint_target=np.full(7, tick)))
    # Shape/dtype/column errors must leave all already written columns intact.
    before = {key: len(value) for key, value in log.file["commands"].items()}
    for bad in (dict(time_s=0.0, joint_target=np.zeros(8, int)), dict(time_s=0.0)):
        with pytest.raises(ValueError, match="changed_rollout_column"):
            log.append("commands", bad)
        assert {key: len(value) for key, value in log.file["commands"].items()} == before
    log.finish(dict(passed=False, termination="fixed_diagnostic_horizon"))
    log.close()
    with h5py.File(path) as h5:
        assert h5.attrs["complete"]
        assert len(h5["commands/time_s"]) == 132
        np.testing.assert_array_equal(h5["commands/joint_target"][:, 0], np.arange(1, 133))
        for index, tick in enumerate(range(0, 133, 6)):
            for key in ("rgb", "depth_m", "valid_depth"):
                assert h5[f"captures/{key}"][index].tobytes() == sensor(tick)[key].tobytes()


def test_buffered_numeric_rows_own_inputs_and_flush_on_close(tmp_path):
    path = tmp_path / "buffered-partial.hdf5"
    log = writer(path)
    log.observe(sensor(0), observation(0, 0), {})
    for tick in range(1, 63):
        log.observe(sensor(tick), observation(tick, tick - tick % 6), {})
        target = np.full(7, tick, dtype=float)
        log.command(dict(time_s=(tick - 1) / 60, joint_target=target))
        target[:] = -999  # The next producer update cannot rewrite pending evidence.
        if tick == 59:
            assert len(log.file["control/time_s"]) == 60
    log.close()
    log.close()
    with h5py.File(path) as h5:
        assert not h5.attrs["complete"]
        assert len(h5["control/time_s"]) == 63
        np.testing.assert_array_equal(h5["commands/joint_target"][:, 0], np.arange(1, 63))


def test_partial_storage_failure_is_not_retried_or_marked_complete(tmp_path, monkeypatch):
    path = tmp_path / "write-failure.hdf5"
    log = writer(path)
    log.observe(sensor(0), observation(0, 0), {})
    original = h5py.Dataset.__setitem__
    failures = []

    def fail(dataset, key, value):
        if dataset.name == "/control/time_s":
            failures.append(True)
            raise OSError("injected storage failure")
        return original(dataset, key, value)

    monkeypatch.setattr(h5py.Dataset, "__setitem__", fail)
    with pytest.raises(OSError, match="storage failure"):
        log.finish(dict(passed=False, termination="failure"))
    with pytest.raises(RuntimeError, match="rollout_write_failed"):
        log.flush()
    with pytest.raises(RuntimeError, match="rollout_write_failed"):
        log.append("control", dict(time_s=np.float64(1)))
    log.close()
    assert len(failures) == 1
    with h5py.File(path) as h5:
        assert not h5.attrs["complete"]


def test_unstarted_failure_retains_empty_recording_and_cannot_claim_success(tmp_path):
    path = tmp_path / "startup-failure.hdf5"
    log = writer(path)
    with pytest.raises(ValueError, match="empty_failed_initialization"):
        log.finish_unavailable(dict(passed=True, termination="wrong"))
    outcome = dict(
        passed=False, termination="loaded_start_contact_not_reconstructed", learned_commands=0
    )
    log.finish_unavailable(outcome)
    log.close()
    with h5py.File(path) as h5:
        assert not h5.attrs["complete"] and h5.attrs["initial_observation_unavailable"]
        assert json.loads(h5["metadata"].attrs["outcome"]) == outcome
        assert not list(h5["control"]) and not list(h5["commands"])
    log = writer(tmp_path / "observed.hdf5")
    log.observe(sensor(0), observation(0, 0), {})
    with pytest.raises(ValueError, match="empty_failed_initialization"):
        log.finish_unavailable(outcome)
    log.close()
