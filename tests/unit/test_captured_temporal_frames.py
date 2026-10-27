"""Recorded CUDA image values remain explicit, portable and mask-gated."""

import json
import shutil

import h5py
import numpy as np
import pytest

from alexdoor_xas.dataset.temporal import TemporalEpisode, prepare_temporal
from alexdoor_xas.policies.temporal import TemporalConfig
from helpers import temporal_command, temporal_recorder


@pytest.fixture
def captured_frames(tmp_path):
    folder = tmp_path / "episode"
    folder.mkdir()
    raw = folder / "episode.hdf5"
    cfg = TemporalConfig(image_size=(2, 3), static_time_s=0, warmup_s=0)
    preprocessing = dict(image_size=[2, 3], depth_scale_m=5.0, depth_interval_m=[0.1, 5.0])
    log = temporal_recorder(raw, image_compression="gzip")
    log.reset(0)
    for row in range(301):
        t = row / 60
        log.state(np.zeros(9), np.zeros(9), acquired=t, available=t, generation=0)
        if row % 6 == 0:
            log.capture(
                np.zeros((4, 6, 3), "u1"),
                np.ones((4, 6, 1), "f4"),
                np.ones((4, 6, 1), bool),
                np.eye(3),
                np.eye(4),
                acquired=t,
                available=t,
                generation=0,
            )
            # File-I/O fixture: precomputed values, with no model/GPU execution.
            log.cache_preprocessed(
                np.full((5, 2, 3), row / 300, "f4"), preprocessing, device="cuda:0"
            )
        log.feedback(np.zeros(7), acquired=t, available=t, healthy=True, generation=0)
        if row < 300:
            temporal_command(log, t)
    log.finish(dict(passed=True, termination="fixture", released=True, stabilized=True))
    log.close()
    return folder, raw, cfg


def test_metadata_only_preparation_and_relative_source_lifetime(captured_frames, monkeypatch):
    import alexdoor_xas.dataset.temporal as module

    def forbidden(*args, **kwargs):
        raise AssertionError("Recorded values must not be preprocessed again")

    monkeypatch.setattr(module, "preprocess_rgbd", forbidden)
    folder, raw, cfg = captured_frames
    path = folder / "prepared.hdf5"
    prepare_temporal(
        raw, path, cfg, device="cpu", use_recorded_frames=True, eligible=np.ones(300, bool)
    )
    with h5py.File(path) as f:
        link = f["images"].get("values", getlink=True)
        assert isinstance(link, h5py.ExternalLink) and link.filename == raw.name
        assert f.attrs["uses_recorded_frames"] and f["images/values"].compression == "gzip"
    first, second = TemporalEpisode(path, cfg), TemporalEpisode(path, cfg)
    selected = second.sample(250)
    indices = selected["image_source_index"]
    np.testing.assert_allclose(selected["obs"]["images"][:, 0, 0, 0], indices / 50)
    first.close()
    assert second.batch([250, 299])["obs"]["images"].shape == (2, 23, 5, 2, 3)
    second.close()
    assert not first.frames.cache and not second.frames.cache
    assert first.frames.values is None and second.frames.values is None
    moved = folder.with_name("moved")
    shutil.move(folder, moved)
    with TemporalEpisode(moved / path.name, cfg) as ep:
        np.testing.assert_array_equal(ep.sample(250)["obs"]["images"], selected["obs"]["images"])
    (moved / raw.name).rename(moved / "unavailable.hdf5")
    with pytest.raises(ValueError, match="restore its raw source"):
        TemporalEpisode(moved / path.name, cfg)


@pytest.mark.parametrize("fault", ["missing", "truncated", "preprocessing"])
def test_rejects_unmatched_captured_frames(captured_frames, fault):
    folder, raw, cfg = captured_frames
    with h5py.File(raw, "r+") as f:
        if fault == "missing":
            del f["preprocessed_images"]
        elif fault == "truncated":
            f["preprocessed_images/values"].resize(50, axis=0)
        else:
            f["preprocessed_images"].attrs["preprocessing"] = json.dumps(dict(image_size=[2, 3]))
    with pytest.raises(ValueError, match="captured preprocessed"):
        prepare_temporal(raw, folder / "invalid.hdf5", cfg, device="cpu", use_recorded_frames=True)


def test_capture_cache_never_grants_eligibility(captured_frames):
    folder, raw, cfg = captured_frames
    prepare_temporal(raw, folder / "diagnostic.hdf5", cfg, device="cpu", use_recorded_frames=True)
    with TemporalEpisode(folder / "diagnostic.hdf5", cfg) as ep:
        assert len(ep.training_rows()) == 0
        assert not ep.file["actions/eligible"][:].any()


def test_cache_requires_one_valid_cuda_result_per_capture(tmp_path):
    log = temporal_recorder(tmp_path / "raw.hdf5")
    log.reset(0)
    args = dict(image_size=[2, 3], depth_scale_m=5, depth_interval_m=[0.1, 5])
    values = np.ones((5, 2, 3), "f4")
    with pytest.raises(ValueError, match="preprocessed frame"):
        log.cache_preprocessed(values, args, device="cuda:0")
    log.state(np.zeros(9), np.zeros(9), acquired=0, available=0, generation=0)
    log.capture(
        np.zeros((4, 6, 3), "u1"),
        np.ones((4, 6, 1), "f4"),
        np.ones((4, 6, 1), bool),
        np.eye(3),
        np.eye(4),
        acquired=0,
        available=0,
        generation=0,
    )
    with pytest.raises(ValueError, match="preprocessed frame"):
        log.cache_preprocessed(values, args, device="cpu")
    log.cache_preprocessed(values, args, device="cuda:0")
    with pytest.raises(ValueError, match="alignment"):
        log.cache_preprocessed(values, args, device="cuda:0")
    log.close()
