"""Qualified temporal ACT data sampling; annotations select rows, never features."""

import json
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np

from alexdoor_xas.dataset.temporal import TemporalEpisode, mixed_temporal_batch
from alexdoor_xas.policies.act.temporal import normalize_temporal_batch
from alexdoor_xas.policies.temporal import TemporalConfig


def build_training_index(handoff, destination):
    """Index the admitted union in place, including narrower historical controls."""
    handoff = json.loads(Path(handoff).read_text())
    selected = handoff["all_usable_train"] + handoff["new_development"]
    records, seen = [], set()
    for source in selected:
        path = Path(source["prepared"])
        if path in seen:
            continue
        seen.add(path)
        with h5py.File(path, "r") as file:
            metadata = json.loads(file.attrs["episode"])
        cfg = TemporalConfig(**metadata["observation_config"])
        with TemporalEpisode(path, cfg, cache_frames=1) as ep:
            rows = ep.training_rows(12)
            with h5py.File(ep.file.attrs["source"], "r") as raw:
                phases = raw["evaluator/phase"][:]
                names = json.loads(raw["metadata"].attrs["episode"])["phase_names"]
                roles = (
                    raw["evaluator/action_role"][:]
                    if "action_role" in raw["evaluator"]
                    else np.zeros(len(phases))
                )
            commands = np.searchsorted(ep.action_times, ep.times[rows] - 1e-9)
            if not np.allclose(ep.action_times[commands], ep.times[rows], atol=1e-8, rtol=0):
                raise ValueError("Annotation clock differs from decision clock")
            strata = {name: rows[phases[commands] == i].tolist() for i, name in enumerate(names)}
            edges = np.flatnonzero(phases[1:] != phases[:-1]) + 1
            near = np.zeros(len(rows), bool)
            for edge in edges:
                near |= np.abs(commands - edge) <= 12
            strata["transition"] = rows[near].tolist()
            # Native role 1 is erroneous injection; role 2 is valid expert correction.
            strata["recovery"] = rows[roles[commands] == 2].tolist()
            strata = {k: v for k, v in strata.items() if v}
            if not strata:
                raise ValueError("No admitted strata")
            records.append(
                dict(
                    path=str(path),
                    config=cfg.to_dict(),
                    episode_id=ep.builder.episode_id,
                    split=ep.metadata["split"],
                    asset_id=ep.metadata["asset_id"],
                    source_group=source["source_group"],
                    rows=rows.tolist(),
                    strata=strata,
                )
            )
        print(
            json.dumps(dict(indexed=source.get("case", str(path)), windows=len(rows))), flush=True
        )
    ids = [r["episode_id"] for r in records]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate source identities")
    train_groups = {r["source_group"] for r in records if r["split"] == "train"}
    dev_groups = {r["source_group"] for r in records if r["split"] == "development"}
    if train_groups & dev_groups:
        raise ValueError("Train/development source-family leakage")
    Path(destination).write_text(json.dumps(dict(records=records), indent=2) + "\n")


class TemporalTrainingSet:
    def __init__(self, manifest, *, split="train", data_root=None):
        if split not in ("train", "development"):
            raise ValueError("Final test remains sealed")
        self.records = [
            r for r in json.loads(Path(manifest).read_text())["records"] if r["split"] == split
        ]
        self.episodes = []
        self.doors = defaultdict(lambda: defaultdict(list))
        for i, record in enumerate(self.records):
            path = Path(record["path"])
            if data_root is not None:
                path = Path(data_root) / path.name
            ep = TemporalEpisode(path, TemporalConfig(**record["config"]), cache_frames=4)
            if ep.metadata["split"] != split or ep.builder.episode_id != record["episode_id"]:
                ep.close()
                raise ValueError("Prepared source differs from admitted index")
            self.episodes.append(ep)
            for name in record["strata"]:
                self.doors[record["asset_id"]][name].append(i)
        if not self.episodes:
            raise ValueError("Empty split")

    def batch(self, seed, *, batch_size=32, stats=None):
        if batch_size < 4 or batch_size % 4:
            raise ValueError("Batch size must be a positive multiple of four")
        rng = np.random.default_rng(seed)
        selections = defaultdict(list)
        doors = list(self.doors)
        for _ in range(batch_size // 4):
            door = self.doors[doors[int(rng.integers(len(doors)))]]
            name = list(door)[int(rng.integers(len(door)))]
            index = int(rng.choice(door[name]))
            selections[index].extend(rng.choice(self.records[index]["strata"][name], 4).tolist())
        batch = mixed_temporal_batch((self.episodes[i], rows) for i, rows in selections.items())
        return normalize_temporal_batch(batch, stats) if stats is not None else batch

    def close(self):
        for episode in self.episodes:
            episode.close()
