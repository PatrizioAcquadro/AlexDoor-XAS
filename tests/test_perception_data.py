"""Causal windows, split isolation, balanced sampling and training-only statistics."""

import json

import h5py
import numpy as np
import pytest

from alexdoor_xas.perception.data import TARGETS, PerceptionWindows

CONFIG = dict(history=4, sample_hz=10)


def cache(root, split, asset, n, value):
    path = asset + ".hdf5"
    with h5py.File(root / path, "w") as h5:
        h5.attrs["complete"] = True
        h5["time_s"] = np.arange(n) / 10
        h5["proprio"] = np.full((n, 18), value, np.float32)
        h5["camera"] = np.tile(np.eye(4), (n, 1, 1))
        h5["features"] = np.arange(n, dtype=np.float32)[:, None, None, None]
        h5["geometry"] = np.zeros((n, 4, 1, 1))
        for key in TARGETS:
            h5["labels/" + key] = np.arange(n, dtype=np.float32)
        h5["labels/phase"] = np.ones(n, np.int8)
    return dict(path=path, asset_id=asset, split=split, handedness="left", condition="nominal")


def test_windows_never_cross_episodes_and_normalization_excludes_dev(tmp_path):
    entries = [
        cache(tmp_path, "train", "a", 6, 1),
        cache(tmp_path, "train", "b", 9, 3),
        cache(tmp_path, "development", "c", 5, 1000),
    ]
    (tmp_path / "index.json").write_text(json.dumps(dict(config=CONFIG, episodes=entries)))
    train = PerceptionWindows(tmp_path, "train", CONFIG)
    dev = PerceptionWindows(tmp_path, "development", CONFIG)
    assert len(train) == 3 + 6
    mean, std = train.normalization()
    np.testing.assert_allclose(mean, 2)
    np.testing.assert_allclose(std, 1)
    with pytest.raises(ValueError, match="only.*train"):
        dev.normalization()
    with pytest.raises(ValueError, match="train/development"):
        PerceptionWindows(tmp_path, "test", CONFIG)
    inputs, labels, episode, end = train[3]
    assert (episode, end) == (1, 3)
    np.testing.assert_array_equal(inputs["features"].numpy().ravel(), [0, 1, 2, 3])
    assert set(inputs) == {"features", "geometry", "proprio", "camera"}
    assert set(labels) == set(TARGETS)
    weights = train.weights().numpy()
    assert weights[:3].sum() == pytest.approx(weights[3:].sum())
    with h5py.File(tmp_path / "b.hdf5", "r+") as h5:
        h5["time_s"][4] = 100
    with pytest.raises(ValueError, match="missing or noncausal"):
        PerceptionWindows(tmp_path, "train", CONFIG)


def test_fit_subset_filters_before_windows_and_normalization(tmp_path):
    entries = [
        cache(tmp_path, "train", "left", 6, 1),
        cache(tmp_path, "train", "right", 9, 3),
        cache(tmp_path, "train", "unused", 5, 1000),
        cache(tmp_path, "development", "dev", 5, 2000),
    ]
    entries[1]["handedness"] = "right"
    (tmp_path / "index.json").write_text(json.dumps(dict(config=CONFIG, episodes=entries)))
    subset = PerceptionWindows(tmp_path, "train", CONFIG, asset_ids=["left", "right"])
    assert len(subset) == 9
    assert {e["asset_id"] for e in subset.episodes} == {"left", "right"}
    np.testing.assert_allclose(subset.normalization()[0], 2)
    for selection in (["left", "dev"], ["left", "absent"], ["left", "left"]):
        with pytest.raises(ValueError, match="train"):
            PerceptionWindows(tmp_path, "train", CONFIG, asset_ids=selection)
    with pytest.raises(ValueError, match="train"):
        PerceptionWindows(tmp_path, "development", CONFIG, asset_ids=["dev"])


def test_inspection_is_past_only_and_retained_after_recent_window_moves(tmp_path):
    config = dict(CONFIG, inspection=dict(sample_times_s=[0.1, 0.3]))
    entry = cache(tmp_path, "train", "a", 12, 1)
    with h5py.File(tmp_path / entry["path"], "r+") as h5:
        h5.attrs["metadata"] = json.dumps(dict(inspection=config["inspection"]))
        h5["labels/phase"][:4] = 5
    (tmp_path / "index.json").write_text(json.dumps(dict(config=config, episodes=[entry])))
    dataset = PerceptionWindows(tmp_path, "train", config)
    assert dataset.windows[0] == (0, 7)
    np.testing.assert_array_equal(dataset[0][0]["features"].numpy().ravel(), [1, 3, 4, 5, 6, 7])
    np.testing.assert_array_equal(dataset[-1][0]["features"].numpy().ravel(), [1, 3, 8, 9, 10, 11])
    with h5py.File(tmp_path / entry["path"], "r+") as h5:
        h5["labels/phase"][3] = 0
    with pytest.raises(ValueError, match="inspection observation"):
        PerceptionWindows(tmp_path, "train", config)
