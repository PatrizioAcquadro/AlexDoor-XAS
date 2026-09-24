"""Train-split action and observation normalization statistics."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .loader import EpisodeDataset, obs_matrix, validate_obs_keys
from .splits import _safe_view_id

_STD_FLOOR = 1e-8
_NORM_STATS_FILENAME = "norm_stats.json"


@dataclass(frozen=True)
class NormStats:
    """Per-dimension statistics of one quantity."""

    mean: np.ndarray
    std: np.ndarray
    min: np.ndarray
    max: np.ndarray
    count: int

    def validate(self) -> None:
        arrays = (self.mean, self.std, self.min, self.max)
        if (
            self.mean.ndim != 1
            or not self.mean.size
            or any(a.shape != self.mean.shape for a in arrays)
        ):
            raise ValueError("normalization arrays must be matching non-empty vectors")
        if not all(np.isfinite(a).all() for a in arrays):
            raise ValueError("normalization arrays must be finite")
        if (self.std < _STD_FLOOR).any() or (self.min > self.max).any() or self.count <= 0:
            raise ValueError("normalization requires positive count/std and min <= max")

    @classmethod
    def from_rows(cls, arrays: list[np.ndarray]) -> NormStats:
        if not arrays:
            raise ValueError("cannot compute stats from an empty list")
        rows = [np.asarray(array, dtype=np.float64) for array in arrays]
        if any(array.ndim != 2 or array.shape[0] == 0 for array in rows):
            raise ValueError("normalization inputs must all be non-empty (N, D) arrays")
        if len({array.shape[1] for array in rows}) != 1:
            raise ValueError("normalization inputs have inconsistent feature dimensions")
        if not all(np.isfinite(array).all() for array in rows):
            raise ValueError("normalization inputs must be finite")
        stacked = np.concatenate(rows)
        return cls(
            mean=stacked.mean(axis=0),
            std=np.maximum(stacked.std(axis=0), _STD_FLOOR),
            min=stacked.min(axis=0),
            max=stacked.max(axis=0),
            count=int(stacked.shape[0]),
        )

    @property
    def dim(self) -> int:
        return int(self.mean.shape[0])

    def normalize(self, x: np.ndarray) -> np.ndarray:
        return (np.asarray(x, dtype=np.float64) - self.mean) / self.std

    def denormalize(self, x: np.ndarray) -> np.ndarray:
        return np.asarray(x, dtype=np.float64) * self.std + self.mean

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "mean": self.mean.tolist(),
            "std": self.std.tolist(),
            "min": self.min.tolist(),
            "max": self.max.tolist(),
            "count": self.count,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> NormStats:
        try:
            result = cls(
                mean=np.asarray(data["mean"], dtype=np.float64),
                std=np.asarray(data["std"], dtype=np.float64),
                min=np.asarray(data["min"], dtype=np.float64),
                max=np.asarray(data["max"], dtype=np.float64),
                count=int(data["count"]),
            )
            result.validate()
            return result
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"invalid normalization block: {error}") from error


@dataclass(frozen=True)
class DatasetNormStats:
    """Train-split statistics required by training and checkpoints."""

    action: NormStats
    obs: NormStats
    obs_keys: tuple[str, ...]
    train_episode_ids: tuple[str, ...]
    action_space: str
    view_id: str | None = None

    def __post_init__(self):
        object.__setattr__(self, "obs_keys", validate_obs_keys(self.obs_keys))

    def to_dict(self) -> dict[str, Any]:
        return {
            "action": self.action.to_dict(),
            "obs": self.obs.to_dict(),
            "obs_keys": list(self.obs_keys),
            "train_episode_ids": list(self.train_episode_ids),
            "action_space": self.action_space,
            "view_id": self.view_id,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> DatasetNormStats:
        try:
            return cls(
                action=NormStats.from_dict(payload["action"]),
                obs=NormStats.from_dict(payload["obs"]),
                obs_keys=validate_obs_keys(payload["obs_keys"]),
                train_episode_ids=tuple(payload["train_episode_ids"]),
                action_space=payload["action_space"],
                view_id=payload.get("view_id"),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"invalid normalization artifact: {error}") from error


def compute_norm_stats(
    dataset: EpisodeDataset,
    train_episode_ids: list[str],
    obs_keys: tuple[str, ...],
    *,
    view_id: str | None = None,
) -> DatasetNormStats:
    obs_keys = validate_obs_keys(obs_keys)
    records = [dataset.by_id(episode_id) for episode_id in train_episode_ids]
    if not records:
        raise ValueError("train split is empty")
    return DatasetNormStats(
        action=NormStats.from_rows([record.actions for record in records]),
        obs=NormStats.from_rows([obs_matrix(record, obs_keys) for record in records]),
        obs_keys=obs_keys,
        train_episode_ids=tuple(train_episode_ids),
        action_space=dataset.action_space,
        view_id=view_id,
    )


def norm_stats_path(dataset_dir: str | Path) -> Path:
    return Path(dataset_dir) / _NORM_STATS_FILENAME


def view_norm_stats_path(dataset_dir: str | Path, view_id: str) -> Path:
    return Path(dataset_dir) / "views" / _safe_view_id(view_id) / _NORM_STATS_FILENAME


def save_norm_stats(path: str | Path, stats: DatasetNormStats) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(json.dumps(stats.to_dict(), indent=2, sort_keys=True) + "\n")
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    return target


def load_norm_stats(path: str | Path) -> DatasetNormStats:
    """Read normalization values and their training membership."""

    payload = json.loads(Path(path).read_text())
    if not isinstance(payload, dict):
        raise ValueError("normalization artifact must be a JSON object")
    return DatasetNormStats.from_dict(payload)


def validate_norm_stats(
    stats: DatasetNormStats,
    dataset: EpisodeDataset,
    train_episode_ids: list[str],
    obs_keys: tuple[str, ...],
    *,
    view_id: str | None = None,
) -> list[str]:
    """Validate schema-level compatibility and recompute every statistic."""
    obs_keys = validate_obs_keys(obs_keys)
    errors: list[str] = []
    if stats.action_space != dataset.action_space:
        errors.append(
            f"norm stats action_space {stats.action_space!r} != dataset {dataset.action_space!r}"
        )
    if stats.train_episode_ids != tuple(train_episode_ids):
        errors.append("norm stats train_episode_ids do not match the train split")
    if stats.obs_keys != obs_keys:
        errors.append(f"norm stats obs_keys {stats.obs_keys!r} != {obs_keys!r}")
    if stats.view_id != view_id:
        errors.append(f"norm stats view_id {stats.view_id!r} != {view_id!r}")
    expected_obs_dim = obs_matrix(dataset[0], obs_keys).shape[1]
    if stats.action.dim != dataset.action_dim:
        errors.append(f"norm stats action dim {stats.action.dim} != dataset {dataset.action_dim}")
    if stats.obs.dim != expected_obs_dim:
        errors.append(f"norm stats obs dim {stats.obs.dim} != selected fields {expected_obs_dim}")
    for name, block in (("action", stats.action), ("obs", stats.obs)):
        try:
            block.validate()
        except ValueError as error:
            errors.append(f"norm stats {name}: {error}")
    try:
        recomputed = compute_norm_stats(dataset, train_episode_ids, obs_keys, view_id=view_id)
    except (IndexError, KeyError, TypeError, ValueError) as error:
        errors.append(f"normalization numerical recomputation failed: {error}")
    else:
        for name in ("action", "obs"):
            stored = getattr(stats, name)
            expected = getattr(recomputed, name)
            if stored.count != expected.count:
                errors.append(f"norm stats recomputed {name} count mismatch")
            for field in ("mean", "std", "min", "max"):
                if not np.array_equal(getattr(stored, field), getattr(expected, field)):
                    errors.append(f"norm stats recomputed {name} {field} mismatch")
    return errors
