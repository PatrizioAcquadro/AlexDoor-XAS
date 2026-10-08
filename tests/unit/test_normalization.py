"""Normalization rejects invalid observations and preserves constant dimensions."""

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from alexdoor_xas.dataset.normalize import NormStats
from alexdoor_xas.dataset.observations import obs_matrix, validate_obs_keys


def test_normalization_roundtrip_and_invalid_statistics():
    rows = np.array([[1.0, 4.0], [3.0, 4.0]])
    stats = NormStats.from_rows([rows[:1], rows[1:]])
    restored = NormStats.from_dict(stats.to_dict())
    np.testing.assert_array_equal(restored.denormalize(restored.normalize(rows)), rows)
    assert np.isfinite(restored.normalize(rows)).all()
    for bad in (
        replace(stats, std=np.zeros(2)),
        replace(stats, count=0),
        replace(stats, mean=np.array([np.nan, 0])),
        replace(stats, max=np.zeros(2)),
    ):
        with pytest.raises(ValueError):
            bad.validate()


@pytest.mark.parametrize("keys", [(), "position", ("position", "position")])
def test_observation_order_must_be_explicit_and_unique(keys):
    with pytest.raises(ValueError):
        validate_obs_keys(keys)


def test_observations_reject_missing_or_invalid_fields():
    record = SimpleNamespace(episode_id="b1", n_steps=2, obs={"position": np.ones((2, 3))})
    for key in ("door_angle_rad", "contact.sensed"):
        with pytest.raises(ValueError, match="missing"):
            obs_matrix(record, (key,))
    for value in (np.ones((1, 3)), np.array([[0.0], [np.nan]])):
        record.obs["position"] = value
        with pytest.raises(ValueError, match="invalid"):
            obs_matrix(record, ("position",))
