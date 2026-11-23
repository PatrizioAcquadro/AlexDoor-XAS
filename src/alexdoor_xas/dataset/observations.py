"""Ordered observed fields shared by B1 sampling and checkpoint validation."""

import numpy as np

from .storage import ObservationArray


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


def canonical_matrix(record, keys):
    """Return the shared matrix only while every selected column still aliases it."""
    matrix = getattr(record, "observation_matrix", None)
    if matrix is None or tuple(record.obs) != keys or matrix.shape[0] != record.n_steps:
        return None
    offset = 0
    for key in keys:
        column = record.obs[key]
        if column.ndim != 2:
            return None
        end = offset + column.shape[1]
        if isinstance(matrix, ObservationArray):
            if not matrix.view(columns=slice(offset, end)).same_view(column):
                return None
        else:
            expected = matrix[:, offset:end]
            if (
                column.shape != expected.shape
                or column.strides != expected.strides
                or column.__array_interface__["data"][0] != expected.__array_interface__["data"][0]
            ):
                return None
        offset = end
    return matrix if offset == matrix.shape[1] else None


def obs_matrix(record, obs_keys: tuple[str, ...], *, rows=None) -> np.ndarray:
    """Select rows before converting fields; preserve a canonical NumPy view."""
    keys = validate_obs_keys(obs_keys)
    matrix = canonical_matrix(record, keys)
    if matrix is not None:
        selected = (
            matrix[:]
            if rows is None and isinstance(matrix, ObservationArray)
            else (matrix if rows is None else matrix[rows])
        )
        if not all(np.isfinite(selected[i : i + 64]).all() for i in range(0, len(selected), 64)):
            raise ValueError(f"episode {record.episode_id}: invalid observed features")
        return selected
    columns = []
    for key in keys:
        if key not in record.obs:
            raise ValueError(f"episode {record.episode_id}: missing proprioceptive field {key!r}")
        values = record.obs[key]
        if np.ndim(values) == 0 or values.shape[0] != record.n_steps:
            raise ValueError(f"episode {record.episode_id}: invalid proprioceptive field {key!r}")
        array = np.asarray(values if rows is None else values[rows], dtype=np.float64)
        if not all(np.isfinite(array[i : i + 64]).all() for i in range(0, len(array), 64)):
            raise ValueError(f"episode {record.episode_id}: invalid proprioceptive field {key!r}")
        columns.append(array.reshape(len(array), -1))
    return np.concatenate(columns, axis=1)
