"""Complete physical contact volumes stay inside finite observed material at receipt."""

from dataclasses import replace

import numpy as np
import pytest

from alexdoor_xas.action.frames import ObjectFrame
from alexdoor_xas.action.spaces import A1_JOINT_DELTA, A2_EE_DELTA
from alexdoor_xas.perception.admission import require_current_admission
from alexdoor_xas.perception.local_admission import MaterialMotionSupport, admit_local_action
from perception_helpers import local_inputs


def finite_grid(patch, half=0.125, cell=0.015625, radius=0.01):
    origin = patch.world_pose.origin
    lower = origin[1:] - half
    axis = np.arange(round(2 * half / cell) + 1) * cell
    a, b = np.meshgrid(axis, axis)
    clearance = np.minimum.reduce([a, b, 2 * half - a, 2 * half - b])
    return MaterialMotionSupport(
        patch.world_pose,
        ((0.0, 0.0), (1.0, 0.0), (0.0, 1.0)),
        tuple(lower),
        cell,
        clearance,
        ((*origin, radius),),
        (0.0, 0.0),
    )


@pytest.mark.parametrize("space", [A1_JOINT_DELTA, A2_EE_DELTA])
def test_complete_volume_replaces_nominal_robot_reserve_only_with_independent_sweep(space):
    estimate, proposal, evidence, limits = local_inputs(space)
    cert = finite_grid(estimate.local.patches[0])
    limits = replace(limits, robot_position_bound_m=1.0, stop_travel_m=1.0)
    assert not admit_local_action(estimate, proposal, evidence, limits, 1, generation=0).admitted
    evidence = replace(evidence, material_motion_support=cert, swept_collision_margin_m=0.01)
    result = admit_local_action(estimate, proposal, evidence, limits, 1, generation=0)
    assert result.admitted
    assert result.decision["margin_kind"] == "complete_physical_material_motion"
    require_current_admission(result, estimate, proposal.times_s[-1])
    for invalid in [
        replace(evidence, swept_collision_margin_m=None),
        replace(evidence, swept_collision_margin_m=0.0),
        replace(evidence, finger_clearance_m=(None, 0.04)),
    ]:
        assert not admit_local_action(estimate, proposal, invalid, limits, 1, generation=0).admitted


@pytest.mark.parametrize("perturbation", ["shift", "uncertainty", "hole", "outside"])
def test_receipt_and_admission_keep_holes_entire_spheres_and_new_pose(perturbation):
    estimate, proposal, evidence, limits = local_inputs()
    patch = estimate.local.patches[0]
    cert = finite_grid(patch)
    evidence = replace(
        evidence,
        material_motion_support=cert,
        swept_collision_margin_m=0.01,
        response_position_bound_m=0.3,
    )
    result = admit_local_action(estimate, proposal, evidence, limits, 1, generation=0)
    assert result.admitted
    if perturbation in ("hole", "outside"):
        if perturbation == "hole":
            grid = cert.clearance_m.copy()
            grid[8, 8] = 0
            cert = replace(cert, clearance_m=grid)
        else:
            cert = replace(cert, spheres=((1.0, 1.0, 1.1, 0.01),))
        failure = admit_local_action(
            estimate,
            proposal,
            replace(evidence, material_motion_support=cert),
            limits,
            1,
            generation=0,
        )
        assert not failure.admitted and failure.reason == "material_motion_support_exhausted"
    else:
        changed = (
            replace(
                patch,
                world_pose=replace(
                    patch.world_pose, origin=patch.world_pose.origin + [0, 0.115, 0]
                ),
            )
            if perturbation == "shift"
            else replace(patch, pose_support=replace(patch.pose_support, position_bound_m=0.12))
        )
        current = replace(estimate, local=replace(estimate.local, patches=(changed,)))
        with pytest.raises(ValueError, match="material_motion_support_exhausted"):
            require_current_admission(result, current, proposal.times_s[-1])


def test_boundary_and_motion_are_not_fitted_tolerances():
    estimate, _, _, _ = local_inputs()
    patch = estimate.local.patches[0]
    zero = replace(patch.pose_support, position_bound_m=0.0, rotation_bound_rad=0.0)
    patch = replace(
        patch,
        world_pose=ObjectFrame(np.zeros(3), np.eye(3)),
        geometry_support=zero,
        pose_support=zero,
    )
    cert = finite_grid(patch, radius=0.125)
    assert cert.margin(patch) == 0
    with pytest.raises(ValueError, match="material_motion_support_exhausted"):
        cert.require(patch)
    assert replace(cert, spheres=((0.0, 0.0, 0.0, 0.12499),)).require(patch) > 0
    assert replace(cert, spheres=((0.0, 0.0, 0.0, 0.12501),)).margin(patch) < 0
    cert = replace(cert, spheres=((0.0, 0.0, 0.0, 0.01),), interval_motion_m_rad=(0.005, 0.01))
    assert cert.margin(patch, remaining_fraction=0) - cert.margin(patch) == pytest.approx(0.0051)


@pytest.mark.parametrize(
    "changes",
    [
        dict(spheres=()),
        dict(cell_m=0),
        dict(spheres=((1.0, 0.0, 1.1, -0.01),)),
        dict(interval_motion_m_rad=(-0.1, 0)),
        dict(geometry_position_prepaid_m=0.1),
    ],
)
def test_missing_or_invalid_volume_does_not_supply_support(changes):
    estimate, proposal, evidence, limits = local_inputs()
    cert = replace(finite_grid(estimate.local.patches[0]), **changes)
    result = admit_local_action(
        estimate,
        proposal,
        replace(evidence, material_motion_support=cert, swept_collision_margin_m=0.01),
        limits,
        1,
        generation=0,
    )
    assert not result.admitted


def test_complete_box_avoids_empty_sphere_corners_but_keeps_interior_holes():
    from alexdoor_xas.perception.local_admission import TargetReliefBox

    estimate, _, _, _ = local_inputs()
    patch = estimate.local.patches[0]
    zero = replace(patch.pose_support, position_bound_m=0.0, rotation_bound_rad=0.0)
    patch = replace(
        patch,
        world_pose=ObjectFrame(np.zeros(3), np.eye(3)),
        geometry_support=zero,
        pose_support=zero,
    )
    # Long normal extent has a large sphere but a small complete tangent extent.
    box = TargetReliefBox(
        tuple(map(tuple, np.eye(3))), (-0.2, -0.02, -0.02), (0.2, 0.02, 0.02), 0.005
    )
    cert = replace(finite_grid(patch), spheres=((0.0, 0.0, 0.0, 0.21),), boxes=(box,))
    assert replace(cert, boxes=()).margin(patch) < 0
    assert cert.require(patch) > 0
    grid = cert.clearance_m.copy()
    grid[8, 8] = 0.0
    assert replace(cert, clearance_m=grid).margin(patch) < 0
    wider = replace(box, upper_m=(0.2, 0.125, 0.02))
    assert replace(cert, boxes=(wider,)).margin(patch) < 0
    changed = replace(patch, pose_support=replace(zero, rotation_bound_rad=0.6))
    assert cert.margin(changed) < 0


def test_initial_world_grid_does_not_rotate_again_with_the_shared_object_basis():
    estimate, proposal, evidence, limits = local_inputs()
    patch = estimate.local.patches[0]
    cert = finite_grid(patch)
    # Static local frame confidence does not describe another physical motion
    # of the already measured world grid. Registration error still does.
    uncertain_basis = replace(
        patch, geometry_support=replace(patch.geometry_support, rotation_bound_rad=1.0)
    )
    assert cert.margin(uncertain_basis) == cert.margin(patch)
    uncertain_registration = replace(
        patch, pose_support=replace(patch.pose_support, rotation_bound_rad=1.0)
    )
    assert cert.margin(uncertain_registration) < cert.margin(patch)
    # Exact common-factor cancellation, including a nontrivial zone offset.
    from scipy.spatial.transform import Rotation

    def transform(rot, offset):
        t = np.eye(4)
        t[:3, :3] = Rotation.from_rotvec(rot).as_matrix()
        t[:3, 3] = offset
        return t

    c0 = transform([0.1, 0.2, -0.3], [0.3, -0.4, 0.6])
    ct = transform([-0.2, 0.4, 0.1], [-0.1, 0.3, 0.7])
    motion = transform([0.2, 0.0, 0.5], [0.1, 0.02, -0.03])
    basis = transform([0.8, -0.5, 0.4], [0.4, 0.5, 0.6])
    np.testing.assert_allclose(
        (c0 @ basis) @ np.linalg.inv(ct @ motion @ basis),
        c0 @ np.linalg.inv(motion) @ np.linalg.inv(ct),
        atol=1e-15,
    )
