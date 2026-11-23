"""Sample padded action chunks and deterministic training batches."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from .observations import canonical_matrix, obs_matrix, validate_obs_keys

if TYPE_CHECKING:
    from .b1 import B1Dataset


@dataclass(frozen=True)
class ChunkSample:
    """One training sample: obs at t + the H-step action window from t."""

    obs: np.ndarray  # (obs_dim,)
    actions: np.ndarray  # (H, D), zero-padded past the episode end
    is_pad: np.ndarray  # (H,) bool, True where the action is padding


class ChunkSampler:
    """Enumerate every (episode, tick) sample of a split at a fixed horizon."""

    def __init__(
        self,
        dataset: B1Dataset,
        horizon: int,
        obs_keys: tuple[str, ...],
        episode_ids: list[str] | None = None,
    ):
        if horizon < 1:
            raise ValueError(f"horizon must be >= 1, got {horizon}")
        self.dataset = dataset
        self.horizon = horizon
        self.obs_keys = validate_obs_keys(obs_keys)
        ids = dataset.episode_ids if episode_ids is None else episode_ids
        self._records = [dataset.by_id(episode_id) for episode_id in ids]
        if not self._records:
            raise ValueError("sampler has no episodes")
        # Keep aliases for contiguous legacy matrices; never gather a corpus.
        self._obs = [canonical_matrix(record, self.obs_keys) for record in self._records]
        first_rows = [
            obs_matrix(record, self.obs_keys, rows=slice(0, 1)) for record in self._records
        ]
        self._obs_dim = first_rows[0].shape[1]
        self._obs_dtype = np.result_type(*(row.dtype for row in first_rows))
        self._index: list[tuple[int, int]] = [
            (rec_idx, t)
            for rec_idx, record in enumerate(self._records)
            for t in range(record.n_steps)
        ]

    @property
    def obs_dim(self) -> int:
        return self._obs_dim

    @property
    def action_dim(self) -> int:
        return self._records[0].action_dim

    def __len__(self) -> int:
        return len(self._index)

    def sample(self, index: int) -> ChunkSample:
        rec_idx, t = self._index[index]
        record = self._records[rec_idx]
        window = record.actions[t : t + self.horizon]
        n_valid = window.shape[0]
        actions = np.zeros((self.horizon, record.action_dim), dtype=np.float64)
        actions[:n_valid] = window
        is_pad = np.arange(self.horizon) >= n_valid
        return ChunkSample(
            obs=obs_matrix(record, self.obs_keys, rows=slice(t, t + 1))[0],
            actions=actions,
            is_pad=is_pad,
        )

    def batch(self, indices):
        """Group reads by episode, restoring the original seeded sample order."""
        locations = [self._index[int(index)] for index in indices]
        obs = np.empty((len(locations), self.obs_dim), dtype=self._obs_dtype)
        actions = np.zeros((len(locations), self.horizon, self.action_dim), dtype=np.float64)
        is_pad = np.ones((len(locations), self.horizon), dtype=bool)
        for rec_idx in dict.fromkeys(record for record, _ in locations):
            slots = [i for i, (record, _) in enumerate(locations) if record == rec_idx]
            ticks = np.array([locations[i][1] for i in slots])
            record = self._records[rec_idx]
            obs[slots] = obs_matrix(record, self.obs_keys, rows=ticks)
            for slot, tick in zip(slots, ticks, strict=True):
                window = record.actions[tick : tick + self.horizon]
                actions[slot, : len(window)] = window
                is_pad[slot, : len(window)] = False
        return dict(obs=obs, actions=actions, is_pad=is_pad)


class BatchIterator:
    """Seeded single-pass batches of ``obs``, ``actions``, and ``is_pad``."""

    def __init__(
        self,
        sampler: ChunkSampler,
        batch_size: int,
        seed: int = 0,
        drop_last: bool = False,
        sample_groups: np.ndarray | None = None,
    ):
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}")
        self.sampler = sampler
        self.batch_size = batch_size
        self.seed = seed
        self.drop_last = drop_last
        self.groups = None
        if sample_groups is not None:
            labels = np.asarray(sample_groups)
            if labels.shape != (len(sampler),) or labels.dtype.kind not in "iu":
                raise ValueError("Sample groups require one integer label per chunk")
            self.groups = [np.flatnonzero(labels == value) for value in np.unique(labels)]
            if not self.groups:
                raise ValueError("Sample groups cannot be empty")

    def __iter__(self):
        rng = np.random.default_rng(self.seed)
        if self.groups is None:
            order = rng.permutation(len(self.sampler))
        else:
            # Equal group exposure (within one sample), with replacement inside
            # each group. No labels are concatenated into policy observations.
            assignments = rng.permutation(np.arange(len(self.sampler)) % len(self.groups))
            order = np.empty(len(self.sampler), dtype=np.int64)
            for group, members in enumerate(self.groups):
                slots = np.flatnonzero(assignments == group)
                order[slots] = rng.choice(members, size=len(slots), replace=True)
        for start in range(0, len(order), self.batch_size):
            chosen = order[start : start + self.batch_size]
            if self.drop_last and len(chosen) < self.batch_size:
                return
            if isinstance(self.sampler, ChunkSampler):
                yield self.sampler.batch(chosen)
                continue
            samples = [self.sampler.sample(int(i)) for i in chosen]
            yield {
                "obs": np.stack([s.obs for s in samples]),
                "actions": np.stack([s.actions for s in samples]),
                "is_pad": np.stack([s.is_pad for s in samples]),
            }

    def __len__(self) -> int:
        n_batches, remainder = divmod(len(self.sampler), self.batch_size)
        return n_batches if (self.drop_last or remainder == 0) else n_batches + 1


def collate_torch(batch: dict[str, Any]):
    """Convert a numpy batch to float32 torch tensors (torch is optional)."""
    import torch

    def convert(value):
        if isinstance(value, dict):
            return {key: convert(item) for key, item in value.items()}
        if isinstance(value, np.ndarray):
            dtype = torch.float32 if value.dtype.kind == "f" else None
            return torch.as_tensor(value, dtype=dtype)
        return value

    return convert(batch)
