"""Complete/failure/interruption and requested versus applied command semantics."""

import json

import h5py
import numpy as np
import pytest

from helpers import temporal_command as record_command
from helpers import temporal_recorder as recorder


def test_failed_episode_native_clocks_requests_and_lossless_images(tmp_path):
    path = tmp_path / "failure.hdf5"
    log = recorder(path)
    log.reset(0)
    depth = np.ones((4, 6, 1), "f4")
    depth.view("u4")[0, 0, 0] = 0x7FC00017
    log.state(np.zeros(9), np.zeros(9), acquired=0, available=0, generation=0)
    log.capture(
        np.zeros((4, 6, 3), "u1"),
        depth,
        np.isfinite(depth),
        np.eye(3),
        np.eye(4),
        acquired=0,
        available=0,
        generation=0,
    )
    record_command(log, 0)
    with pytest.raises(ValueError, match="pre-state"):
        record_command(log, 0)
    log.feedback(np.zeros(7), acquired=1 / 120, available=1 / 120, healthy=True, generation=0)
    log.state(np.zeros(9), np.zeros(9), acquired=1 / 60, available=1 / 60, generation=0)
    record_command(log, 1 / 60, accepted=False)
    log.state(np.zeros(9), np.zeros(9), acquired=2 / 60, available=2 / 60, generation=0)
    log.finish(dict(passed=False, termination="safe_refusal"))
    log.close()
    with h5py.File(path) as file:
        assert file.attrs["complete"]
        assert not json.loads(file["metadata"].attrs["outcome"])["passed"]
        np.testing.assert_array_equal(file["captures/depth_m"][0].view("u4"), depth.view("u4"))
        np.testing.assert_array_equal(file["commands/requested_target"][0], 1)
        np.testing.assert_array_equal(file["commands/applied_target"][0], 0.5)
        assert not file["commands/accepted"][1]
        assert len(file["control/acquired"]) == 3 and len(file["captures/acquired"]) == 1


def test_interruption_flushes_without_certifying_completion(tmp_path):
    path = tmp_path / "interrupted.hdf5"
    log = recorder(path)
    log.reset(0)
    log.state(np.zeros(9), np.zeros(9), acquired=0, available=0, generation=0)
    record_command(log, 0)
    with pytest.raises(ValueError, match="terminal"):
        log.finish(dict(passed=False, termination="interrupted"))
    log.close()
    with h5py.File(path) as file:
        assert not file.attrs["complete"]
        assert len(file["commands/dispatch_s"]) == 1
    with pytest.raises(FileExistsError):
        recorder(path)
