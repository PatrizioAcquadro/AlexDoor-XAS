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


def obs_matrix(record, obs_keys: tuple[str, ...], *, rows=None) -> np.ndarray:
    """Read selected observed rows, retaining a canonical matrix view when possible."""
    keys = validate_obs_keys(obs_keys)
    matrix = getattr(record, "observation_matrix", None)
    if matrix is not None and tuple(record.obs) == keys:
        offset, canonical = 0, matrix.shape[0] == record.n_steps
        for key in keys:
            column = np.asanyarray(record.obs[key])
            if column.ndim != 2:
                canonical = False
                break
            expected = matrix[:, offset : offset + column.shape[1]]
            canonical &= (
                column.shape == expected.shape
                and column.strides == expected.strides
                and (
                    column.__array_interface__["data"][0] == expected.__array_interface__["data"][0]
                )
            )
            offset += column.shape[1]
        if canonical and offset == matrix.shape[1]:
            selected = matrix if rows is None else matrix[rows]
            if not all(
                np.isfinite(selected[i : i + 64]).all() for i in range(0, len(selected), 64)
            ):
                raise ValueError(f"episode {record.episode_id}: invalid observed features")
            return selected
    columns = []
    for key in keys:
        if key not in record.obs:
            raise ValueError(f"episode {record.episode_id}: missing proprioceptive field {key!r}")
        values = np.asanyarray(record.obs[key])
        if values.ndim == 0 or values.shape[0] != record.n_steps:
            raise ValueError(f"episode {record.episode_id}: invalid proprioceptive field {key!r}")
        array = np.asarray(values if rows is None else values[rows], dtype=np.float64)
        if not all(np.isfinite(array[i : i + 64]).all() for i in range(0, len(array), 64)):
            raise ValueError(f"episode {record.episode_id}: invalid proprioceptive field {key!r}")
        columns.append(array.reshape(len(array), -1))
    return np.concatenate(columns, axis=1)
