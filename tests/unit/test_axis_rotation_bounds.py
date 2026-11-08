"""Directional inverse transport covers correlated basis errors and finite volumes."""

from dataclasses import replace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from alexdoor_xas.perception.contracts import AxisRotationBounds


def test_two_basis_transport_covers_correlated_errors_and_entire_finite_balls():
    rng = np.random.default_rng(71)
    for trial in range(90):
        initial, current = Rotation.random(2, random_state=rng).as_matrix()
        errors = rng.normal(size=(2, 3)) * 0.03
        # Include correlated errors and rotations around individual measured axes.
        if trial % 3 == 0:
            errors[1] = errors[0]
        elif trial % 3 == 1:
            errors[:, :2] = 0
        true_initial = initial @ Rotation.from_rotvec(errors[0]).as_matrix()
        true_current = current @ Rotation.from_rotvec(errors[1]).as_matrix()
        angle = float(np.linalg.norm(errors, axis=1).sum())
        chord = 2 * np.sin(angle / 2)
        chords = np.minimum(
            chord,
            np.linalg.norm(true_initial - initial, axis=0)
            + np.linalg.norm(true_current - current, axis=0),
        )
        bound = AxisRotationBounds(tuple(map(tuple, current)), tuple(chords), angle)
        centers = rng.normal(size=(7, 3))
        centers[0] = 0
        directions = rng.normal(size=(100, 3))
        directions /= np.linalg.norm(directions, axis=1)[:, None]
        radius = (trial % 4) * 0.03
        delta = true_initial @ true_current.T - initial @ current.T
        for center in centers:
            points = center + radius * directions
            actual = np.linalg.norm(points @ delta.T, axis=1).max()
            assert actual <= bound.displacement_bound([center], radius, angle) + 1e-14
        certified = bound.displacement_bound(centers, radius, angle)
        assert certified <= (np.linalg.norm(centers, axis=1).max() + radius) * chord
        world = Rotation.random(random_state=rng).as_matrix()
        moved = replace(bound, basis_world=tuple(map(tuple, world @ current)))
        assert moved.displacement_bound(centers @ world.T, radius, angle) == pytest.approx(
            certified
        )


@pytest.mark.parametrize(
    "changes",
    [
        dict(basis_world=((1, 0, 0), (0, 2, 0), (0, 0, 1))),
        dict(basis_world=((-1, 0, 0), (0, 1, 0), (0, 0, 1))),
        dict(chord_bounds=(-0.01, 0.01, 0.01)),
        dict(chord_bounds=(0.2, 0.01, 0.01)),
        dict(chord_bounds=(float("nan"), 0.01, 0.01)),
        dict(rotation_bound_rad=-0.1),
    ],
)
def test_invalid_axis_evidence_does_not_supply_precision(changes):
    bound = AxisRotationBounds(tuple(map(tuple, np.eye(3))), (0.01, 0.01, 0.01), 0.1)
    with pytest.raises(ValueError, match="invalid_axis_rotation_bound"):
        replace(bound, **changes).displacement_bound([[0, 0, 1]], 0.02, 0.1)
    with pytest.raises(ValueError, match="stale_axis_rotation_bound"):
        bound.displacement_bound([[0, 0, 1]], 0.02, 0.2)
