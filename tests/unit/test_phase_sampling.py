"""Rare train phases get balanced exposure without changing causal observations."""

import runpy
from types import SimpleNamespace

import h5py
import numpy as np
import pytest

from alexdoor_xas.dataset.sampling import BatchIterator, ChunkSample


class IndexedChunks:
    def __len__(self):
        return 1001

    def sample(self, index):
        return ChunkSample(np.array([index]), np.zeros((2, 7)), np.array([False, True]))


def test_balanced_exposure_repeats_seed_and_retains_inputs_masks():
    sampler = IndexedChunks()
    labels = np.repeat(np.arange(3), [900, 100, 1])
    batches = list(BatchIterator(sampler, 64, seed=17, sample_groups=labels))
    indices = np.concatenate([batch["obs"][:, 0] for batch in batches])
    counts = np.bincount(labels[indices])
    assert counts.max() - counts.min() <= 1
    assert counts.min() == 333
    assert all(
        np.array_equal(batch["is_pad"][:, 0], np.zeros(len(batch["obs"]))) for batch in batches
    )
    assert all(batch["obs"].shape[1] == 1 for batch in batches)
    again = list(BatchIterator(sampler, 64, seed=17, sample_groups=labels))
    for a, b in zip(batches, again, strict=True):
        np.testing.assert_array_equal(a["obs"], b["obs"])
    uniform = list(BatchIterator(sampler, 1001, seed=17))[0]["obs"][:, 0]
    np.testing.assert_array_equal(uniform, np.random.default_rng(17).permutation(1001))
    with pytest.raises(ValueError, match="integer label"):
        BatchIterator(sampler, 64, sample_groups=labels[:-1])


def test_phase_join_rejects_development_and_mismatched_clocks(tmp_path):
    import json

    api = runpy.run_path("scripts/pilot_offline.py")
    phases = api["PHASES"]
    labels = np.array([phases.index(p) for p in ["approach", "contact", "push", "hold", "release"]])
    times = 25.2 + np.arange(6) / 60
    record = SimpleNamespace(split="train", asset_id="door", times=times[:-1], starts=np.arange(5))
    data = SimpleNamespace(train_ids=("train",), dataset=SimpleNamespace(by_id=lambda _: record))
    protocol = dict(
        train_sampling="phase_balanced_v1",
        episodes=[dict(episode_id="train", asset_id="door", split="train", directory="door")],
    )
    raw_dir = tmp_path / "door"
    raw_dir.mkdir()
    path = raw_dir / "episode.hdf5"
    with h5py.File(path, "w") as h:
        h.create_group("metadata").attrs["episode"] = json.dumps(
            dict(episode_id="train", asset_id="door", split="train")
        )
        h.create_dataset("commands/time_s", data=times[:-1])
        h.create_dataset("commands/phase", data=labels)
    joined = api["training_sample_groups"](tmp_path, data, protocol)
    np.testing.assert_array_equal(joined["train"], labels)
    record.split = "development"
    with pytest.raises(ValueError, match="eligible train"):
        api["training_sample_groups"](tmp_path, data, protocol)
    record.split = "train"
    record.times = times[:-1] + 0.001
    with pytest.raises((AssertionError, IndexError)):
        api["training_sample_groups"](tmp_path, data, protocol)


def test_sampler_correction_preserves_all_other_continuation_fields():
    extend = runpy.run_path("scripts/pilot_offline.py")["validate_epoch_extension"]
    source = dict(epochs=180, training_seed=17, overfit_updates=0, lr=0.0001)
    target = source | dict(epochs=200, train_sampling="phase_balanced_v1")
    extend(source, target)
    extend(target, target | dict(epochs=220))
    for bad in [
        target | dict(lr=0.0002),
        target | dict(training_seed=18),
        target | dict(train_sampling="arbitrary"),
        target | dict(epochs=180),
    ]:
        with pytest.raises(ValueError):
            extend(source, bad)
    with pytest.raises(ValueError):
        extend(target, target | dict(epochs=220, train_sampling="uniform"))
