"""Batch optimization preserves single-sample semantics and bounded eviction."""

import json

import h5py
import numpy as np
import pytest
from test_temporal_observations import streams

from alexdoor_xas.dataset.temporal import LAYOUT, TemporalEpisode
from alexdoor_xas.policies.temporal import TemporalConfig


def test_order_duplicates_masks_holes_padding_and_prefetch_match_samples(tmp_path):
    cfg = TemporalConfig(image_size=(2, 3), static_time_s=0.5)
    raw = streams(cfg)
    raw["images"]["valid"] = np.ones(len(raw["images"]["acquired"]), bool)
    raw["images"]["valid"][10] = False
    command_rows = np.delete(np.arange(300), 100)  # a real internal hole
    eligible = command_rows != 200
    path = tmp_path / "frames.hdf5"
    with h5py.File(path, "w") as file:
        file.attrs.update(
            layout=LAYOUT,
            complete=True,
            episode=json.dumps(dict(episode_id="fixture", split="train", control_dt=1 / 60)),
            preprocessing=json.dumps(
                dict(image_size=[2, 3], depth_scale_m=5.0, depth_interval_m=[0.1, 5.0])
            ),
        )
        for key, stream in raw.items():
            group = file.create_group(key)
            for name, value in stream.items():
                group.create_dataset(name, data=value)
        file.create_dataset("decision_time", data=raw["q"]["acquired"])
        file.create_dataset("decision_generation", data=raw["q"]["generation"])
        file.create_dataset("actions/time", data=command_rows / 60)
        file.create_dataset("actions/generation", data=np.full(len(command_rows), 2))
        file.create_dataset("actions/delta", data=np.repeat(command_rows[:, None], 7, axis=1) / 100)
        file.create_dataset("actions/eligible", data=eligible)
    with TemporalEpisode(path, cfg, cache_frames=2) as ep:
        rows = [0, 150, 100, 200, 299, 300, 150]
        samples = [ep.sample(row) for row in rows]
        ep.frames.cache.clear()
        before = ep.frames.reads
        batch = ep.batch(rows, diagnostics=True)
        expected_indices = set()
        for row in rows:
            snap, selected = ep.builder.build_support(ep.times[row], ep.generations[row])
            expected_indices.update(selected[snap["obs"]["image_valid"]].tolist())
        assert ep.frames.reads - before == len(expected_indices)
        assert len(ep.frames.cache) <= 2 and ep.frames.evictions > 0
        for key in batch["obs"]:
            np.testing.assert_array_equal(batch["obs"][key], [s["obs"][key] for s in samples])
        for key in ("actions", "is_pad"):
            np.testing.assert_array_equal(batch[key], [s[key] for s in samples])
        assert not samples[2]["supervision_eligible"]  # internal target hole
        assert not samples[3]["supervision_eligible"]  # real rejected command
        assert not samples[5]["supervision_eligible"]  # terminal sensory context
        with pytest.raises(ValueError, match="diagnostic"):
            ep.batch(rows)
        good = [250, 251, 250, 299]
        synchronous = list(ep.batches(good, batch_size=2, prefetch=False))
        prefetched = list(ep.batches(good, batch_size=2, prefetch=True))
        for a, b in zip(synchronous, prefetched, strict=True):
            for key in a["obs"]:
                np.testing.assert_array_equal(a["obs"][key], b["obs"][key])
            np.testing.assert_array_equal(a["actions"], b["actions"])
            np.testing.assert_array_equal(a["is_pad"], b["is_pad"])
