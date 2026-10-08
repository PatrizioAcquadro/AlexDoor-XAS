"""Ordered observed fields shared by B1 sampling and checkpoint validation."""

import numpy as np


def validate_obs_keys(value) -> tuple[str, ...]:
    """Require an explicit, non-empty ordered selection without duplicate fields."""
    if (
        not isinstance(value, (tuple, list))
        or not value
        or any(not isinstance(key, str) or not key for key in value)
        or len(set(value)) != len(value)
    ):
        raise ValueError("obs_keys must be a non-empty ordered list of unique field names")
    return tuple(value)


def obs_matrix(record, obs_keys: tuple[str, ...]) -> np.ndarray:
    """Concatenate explicitly selected proprioception into an ``(N, D)`` matrix."""
    columns = []
    for key in validate_obs_keys(obs_keys):
        if key not in record.obs:
            raise ValueError(f"episode {record.episode_id}: missing proprioceptive field {key!r}")
        array = np.asarray(record.obs[key], dtype=np.float64)
        if array.ndim == 0 or array.shape[0] != record.n_steps or not np.isfinite(array).all():
            raise ValueError(f"episode {record.episode_id}: invalid proprioceptive field {key!r}")
        columns.append(array.reshape(record.n_steps, -1))
    return np.concatenate(columns, axis=1)
