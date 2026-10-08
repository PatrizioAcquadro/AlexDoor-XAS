"""Dataset loading, validation, normalization, and batch factories."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from alexdoor_xas.assets.identity import RobotAssetRef

if TYPE_CHECKING:
    from alexdoor_xas.dataset.b1 import B1Dataset
from alexdoor_xas.dataset.normalize import DatasetNormStats
from alexdoor_xas.dataset.sampling import BatchIterator, ChunkSampler

EPOCH_SEED_STRIDE = 10_000

BatchNormalizer = Callable[[dict[str, Any], DatasetNormStats], dict[str, Any]]


@dataclass(frozen=True)
class PolicyData:
    """Validated dataset inputs for policy training."""

    dataset: B1Dataset
    train_ids: tuple[str, ...]
    val_ids: tuple[str, ...]
    stats: DatasetNormStats
    robot_asset: RobotAssetRef | None

    @property
    def obs_dim(self) -> int:
        return self.stats.obs.dim

    @property
    def action_dim(self) -> int:
        return self.stats.action.dim


def normalize_batch(batch: dict[str, Any], stats: DatasetNormStats) -> dict[str, Any]:
    """Z-score observations and actions."""
    normalized = dict(batch)
    normalized["obs"] = stats.obs.normalize(batch["obs"])
    normalized["actions"] = stats.action.normalize(batch["actions"])
    return normalized


def make_train_factory(
    data: PolicyData,
    chunk_size: int,
    batch_size: int,
    seed: int,
    episode_ids: tuple[str, ...] | None = None,
    normalize: BatchNormalizer = normalize_batch,
):
    """Per-epoch reshuffled, normalized train batches (``TrainBatchFactory``)."""
    ids = list(episode_ids if episode_ids is not None else data.train_ids)
    sampler = ChunkSampler(data.dataset, chunk_size, obs_keys=data.stats.obs_keys, episode_ids=ids)
    drop_last = len(sampler) >= batch_size

    def factory(epoch: int):
        iterator = BatchIterator(
            sampler,
            batch_size,
            seed=seed * EPOCH_SEED_STRIDE + epoch,
            drop_last=drop_last,
        )
        return (normalize(batch, data.stats) for batch in iterator)

    return factory


def make_eval_factory(
    data: PolicyData,
    chunk_size: int,
    batch_size: int,
    seed: int,
    episode_ids: tuple[str, ...],
    normalize: BatchNormalizer = normalize_batch,
):
    """Fixed-order normalized batches over a split (``ValBatchFactory``)."""
    sampler = ChunkSampler(
        data.dataset, chunk_size, obs_keys=data.stats.obs_keys, episode_ids=list(episode_ids)
    )

    def factory():
        iterator = BatchIterator(sampler, batch_size, seed=seed, drop_last=False)
        return (normalize(batch, data.stats) for batch in iterator)

    return factory
