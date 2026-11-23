"""Lossless indexed storage, selective legacy access and bounded file ownership."""

import json

import h5py
import numpy as np
import pytest

from alexdoor_xas.action.b1 import ACTION_DIMS
from alexdoor_xas.dataset.b1 import B1Dataset, export_dataset, index_dataset
from alexdoor_xas.dataset.normalize import compute_norm_stats
from alexdoor_xas.dataset.observations import obs_matrix
from alexdoor_xas.dataset.sampling import BatchIterator, ChunkSampler
from alexdoor_xas.dataset.storage import ObservationArray, ObservationFiles
from alexdoor_xas.policies.common.b1_contract import OBS_KEYS
from helpers import TEST_ROBOT_REF, make_b1_episode


@pytest.mark.parametrize("layout", ["compressed", "virtual"])
def test_legacy_sampling_selects_before_numpy_and_preserves_order(
    tmp_path, b1_binding, layout, monkeypatch
):
    episode = make_b1_episode(b1_binding)
    root = export_dataset(
        [episode], tmp_path / "data", b1_binding, TEST_ROBOT_REF, {"left": "train"}
    )
    path = root / "episode_000000.hdf5"
    with h5py.File(path, "r+") as file:
        values = file["observations"][:]
        del file["observations"]
        if layout == "compressed":
            file.create_dataset(
                "observations", data=values, chunks=(1, values.shape[1]), compression="lzf"
            )
        else:
            file.create_dataset("stored", data=values)
            view = h5py.VirtualLayout(values.shape, dtype=values.dtype)
            view[:] = h5py.VirtualSource(".", "stored", shape=values.shape)
            file.create_virtual_dataset("observations", view)

    def forbidden(*args, **kwargs):
        raise AssertionError("whole observation array converted")

    monkeypatch.setattr(ObservationArray, "__array__", forbidden)
    dataset = B1Dataset(root, next(iter(ACTION_DIMS)))
    record = dataset.records[0]
    ticks = np.array([3, 0, 3, record.n_steps - 1])
    np.testing.assert_array_equal(
        obs_matrix(record, OBS_KEYS, rows=ticks), episode.observations[ticks]
    )
    sampler = ChunkSampler(dataset, 5, OBS_KEYS)
    batch = sampler.batch(ticks)
    for slot, tick in enumerate(ticks):
        sample = sampler.sample(tick)
        for key in ("obs", "actions", "is_pad"):
            np.testing.assert_array_equal(batch[key][slot], getattr(sample, key))
    keys = (OBS_KEYS[-1], OBS_KEYS[0])
    selected = obs_matrix(record, keys, rows=ticks)
    np.testing.assert_array_equal(
        selected, np.concatenate([record.obs[k][ticks] for k in keys], axis=1)
    )
    compute_norm_stats(dataset, dataset.episode_ids, keys)
    dataset.close()
    assert not dataset._files.files
    np.testing.assert_array_equal(sampler.sample(0).obs, episode.observations[0])
    dataset.close()


def test_index_conversion_preserves_every_view_stats_masks_and_raw_links(tmp_path, b1_binding):
    episodes = [make_b1_episode(b1_binding, episode_id=f"train-{i}") for i in range(2)]
    root = export_dataset(
        episodes, tmp_path / "original", b1_binding, TEST_ROBOT_REF, {"left": "train"}
    )
    with h5py.File(root / "episode_000000.hdf5", "r+") as file:
        file.create_dataset("raw/retained", data=np.array([1, np.nan]))
    indexed = index_dataset(root, tmp_path / "indexed")
    assert json.loads((root / "meta.json").read_text()) == json.loads(
        (indexed / "meta.json").read_text()
    )
    with h5py.File(indexed / "episode_000000.hdf5") as file:
        np.testing.assert_array_equal(file["raw/retained"][:], [1, np.nan])
        assert isinstance(file.get("raw", getlink=True), h5py.ExternalLink)
        assert file["observations/indices"].shape[0] == len(episodes[0].times)
    for space in ACTION_DIMS:
        with (
            B1Dataset(root, space) as before,
            B1Dataset(indexed, space, check_observations=True) as after,
        ):
            np.testing.assert_array_equal(
                after.records[0].terminal_observation, before.records[0].terminal_observation
            )
            a, b = (
                list(BatchIterator(ChunkSampler(data, 8, OBS_KEYS), 3, seed=19))
                for data in (before, after)
            )
            for x, y in zip(a, b, strict=True):
                for key in x:
                    np.testing.assert_array_equal(x[key], y[key])
                    assert x[key].dtype == y[key].dtype
            x, y = (
                compute_norm_stats(data, data.episode_ids, OBS_KEYS) for data in (before, after)
            )
            assert x.to_dict() == y.to_dict()
    assert root.exists()


def test_selected_and_full_validation_reject_late_invalid_values(tmp_path, b1_binding):
    episode = make_b1_episode(b1_binding)
    root = export_dataset(
        [episode], tmp_path / "data", b1_binding, TEST_ROBOT_REF, {"left": "train"}
    )
    with h5py.File(root / "episode_000000.hdf5", "r+") as file:
        file["observations"][2, 0] = np.nan
    dataset = B1Dataset(root, next(iter(ACTION_DIMS)))
    with pytest.raises(ValueError, match="invalid observed"):
        ChunkSampler(dataset, 1, OBS_KEYS).sample(2)
    with pytest.raises(ValueError, match="Invalid B1 features"):
        B1Dataset(root, next(iter(ACTION_DIMS)), check_observations=True)
    dataset.close()


def test_file_pool_evicts_handles_and_reopens_after_close(tmp_path):
    files = ObservationFiles()
    handles = []
    for i in range(6):
        path = tmp_path / f"{i}.hdf5"
        with h5py.File(path, "w") as file:
            file["observations"] = np.full((3, 2), i)
        handles.append(files.get(path))
    assert len(files.files) == 4 and not handles[0].id.valid and not handles[1].id.valid
    files.close()
    assert all(not handle.id.valid for handle in handles)
    assert files.get(path)["observations"][0, 0] == 5
    files.close()


def test_native_images_preserve_inexact_values_and_signed_zero_when_widening():
    from types import SimpleNamespace

    from alexdoor_xas.dataset.storage import FrameObservationWriter

    binding = SimpleNamespace(config={"visual_dims": [10, 30], "encoder": "rgbd-conv-v1"})
    rows = np.zeros((3, 58), dtype=np.float64)
    rows[1, 0] = -0.0
    rows[2, 0] = 1.0 / 3.0
    writer = FrameObservationWriter(len(rows), binding)
    for row in rows:
        writer.append(row)
    values = writer.finish()
    actual = values[:]
    np.testing.assert_array_equal(actual.view("u8"), rows.view("u8"))
    with h5py.File(values.path) as source:
        assert source["observations/frames"].dtype == np.dtype("f8")
    values.files.close()
    writer.close()


def test_prefetch_preserves_objects_order_errors_and_bounded_lookahead():
    from alexdoor_xas.policies.common.data import prefetch_batches

    produced = []

    def source():
        for i in range(9):
            produced.append(i)
            yield np.full((2, 3), i)
        raise ValueError("retained input failure")

    iterator = prefetch_batches(source())
    np.testing.assert_array_equal(next(iterator), np.zeros((2, 3)))
    iterator.close()
    assert len(produced) <= 2
    iterator = prefetch_batches(source())
    for i in range(9):
        np.testing.assert_array_equal(next(iterator), np.full((2, 3), i))
    with pytest.raises(ValueError, match="retained input failure"):
        next(iterator)


def test_indexed_pilot_has_lossless_dense_float32_compatibility(tmp_path, b1_binding):
    from alexdoor_xas.action.spaces import A1_JOINT_DELTA, A2_EE_DELTA
    from alexdoor_xas.dataset.b1 import MatchedEpisode, materialize_train_data
    from alexdoor_xas.policies.encoding import LEARNED_RECIPE, pilot_binding

    binding = pilot_binding(b1_binding.config["inspection"], recipe=LEARNED_RECIPE)
    spaces = (A1_JOINT_DELTA, A2_EE_DELTA)
    values = np.ones((3, binding.obs_dim), dtype=np.float32)
    episode = MatchedEpisode(
        "train",
        "left",
        "train",
        np.arange(3) / 60,
        values,
        {space: np.zeros((2, ACTION_DIMS[space])) for space in spaces},
        {space: np.arange(2) for space in spaces},
    )
    original = export_dataset(
        [episode], tmp_path / "source", binding, TEST_ROBOT_REF, {"left": "train"}
    )
    indexed = index_dataset(original, tmp_path / "indexed")
    dense = materialize_train_data(
        [indexed], tmp_path / "dense", normalization_episode_ids=["train"]
    )
    with B1Dataset(dense, A1_JOINT_DELTA) as data:
        assert isinstance(data.records[0].observation_matrix, np.memmap)
        np.testing.assert_array_equal(data.records[0].observation_matrix, values[:-1])
        np.testing.assert_array_equal(data.records[0].terminal_observation, values[-1])
        assert data.normalization_episode_ids == ("train",)
