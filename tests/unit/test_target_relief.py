"""Spatial target proofs remain independent of normal and loaded-contact receipts."""

from dataclasses import replace

import numpy as np
import pytest

from alexdoor_xas.action.frames import ObjectFrame
from alexdoor_xas.action.spaces import A1_JOINT_DELTA, A2_EE_DELTA
from alexdoor_xas.perception.admission import require_current_admission
from alexdoor_xas.perception.local_admission import TargetReliefExclusion, admit_local_action
from perception_helpers import local_inputs


def relief(patch):
    return TargetReliefExclusion(
        patch.world_pose,
        ((0.0, 0.0), (1.0, 0.0), (0.0, 1.0)),
        (-0.02, 1.08),
        (0.02, 1.12),
        ((1.0, 0.045, 1.1, 0.01),),
        (0.0, 0.0),
    )


@pytest.mark.parametrize("space", [A1_JOINT_DELTA, A2_EE_DELTA])
@pytest.mark.parametrize("loaded", [False, True])
def test_receipt_renews_full_spatial_proof_even_when_normal_bounds_do_not_change(space, loaded):
    estimate, proposal, evidence, limits = local_inputs(space)
    patch = replace(
        estimate.local.patches[0], normal_position_bound_m=0.0, normal_rotation_bound_rad=0.0
    )
    estimate = replace(estimate, local=replace(estimate.local, patches=(patch,)))
    proposal = replace(proposal, loaded=loaded)
    evidence = replace(evidence, target_relief_exclusion=relief(patch))
    if not loaded:
        evidence = replace(
            evidence,
            finger_clearance_m=(None, None),
            unloaded_separation_margin_m=0.02,
            unloaded_normal_lever_m=1.0,
            swept_collision_margin_m=0.02,
        )
    receipt = admit_local_action(estimate, proposal, evidence, limits, 1, generation=0)
    assert receipt.admitted
    require_current_admission(receipt, estimate, proposal.times_s[-1])
    changed = replace(
        patch,
        world_pose=replace(patch.world_pose, origin=patch.world_pose.origin + [0.0, 0.008, 0.0]),
    )
    # This tangential response is within the original pose/response gate, and
    # normal uncertainty is unchanged. It still invalidates the spatial proof.
    current = replace(estimate, local=replace(estimate.local, patches=(changed,)))
    with pytest.raises(ValueError, match="target_relief_exclusion_exhausted"):
        require_current_admission(receipt, current, proposal.times_s[-1])


@pytest.mark.parametrize("support", ["geometry_support", "pose_support"])
def test_receipt_refuses_new_full_point_uncertainty(support):
    estimate, proposal, evidence, limits = local_inputs()
    patch = estimate.local.patches[0]
    evidence = replace(evidence, target_relief_exclusion=relief(patch))
    receipt = admit_local_action(estimate, proposal, evidence, limits, 1, generation=0)
    assert receipt.admitted
    changed = replace(patch, **{support: replace(getattr(patch, support), position_bound_m=0.012)})
    current = replace(estimate, local=replace(estimate.local, patches=(changed,)))
    with pytest.raises(ValueError, match="target_relief_exclusion_exhausted"):
        require_current_admission(receipt, current, proposal.times_s[-1])


def test_relief_boundary_is_strict_and_keeps_the_complete_sphere():
    estimate, _, _, _ = local_inputs()
    original = estimate.local.patches[0]
    zero = replace(original.geometry_support, position_bound_m=0.0, rotation_bound_rad=0.0)
    patch = replace(
        original,
        world_pose=ObjectFrame(np.zeros(3), np.eye(3)),
        geometry_support=zero,
        pose_support=zero,
    )
    cert = TargetReliefExclusion(
        patch.world_pose,
        ((0.0, 0.0), (1.0, 0.0), (0.0, 1.0)),
        (-0.125, -0.125),
        (0.125, 0.125),
        ((0.0, 0.375, 0.0, 0.25),),
        (0.0, 0.0),
    )
    assert cert.margin(patch) == 0.0
    with pytest.raises(ValueError, match="exclusion_exhausted"):
        cert.require(patch)
    assert replace(cert, spheres=((0.0, 0.37501, 0.0, 0.25),)).require(patch) > 0
    with pytest.raises(ValueError, match="exclusion_exhausted"):
        replace(cert, spheres=((0.0, 0.37499, 0.0, 0.25),)).require(patch)


def test_interval_motion_is_only_paid_for_the_remaining_interval():
    estimate, _, _, _ = local_inputs()
    patch = estimate.local.patches[0]
    cert = replace(relief(patch), interval_motion_m_rad=(0.001, 0.002))
    lever = np.linalg.norm(np.array(cert.spheres[0][:3]) - patch.world_pose.origin) + 0.01
    budget = 0.001 + 0.002 * lever
    assert cert.margin(patch, remaining_fraction=0) - cert.margin(patch) == pytest.approx(budget)
    assert cert.margin(patch, remaining_fraction=0.5) == pytest.approx(
        cert.margin(patch) + budget / 2
    )


@pytest.mark.parametrize(
    "change",
    [
        {"spheres": ()},
        {"spheres": ((1.0, 0.045, 1.1, -0.01),)},
        {"spheres": ((1.0, np.nan, 1.1, 0.01),)},
        {"lower_m": (0.03, 1.08)},
        {"tangent_basis": ((0.0, 0.0), (2.0, 0.0), (0.0, 1.0))},
        {"interval_motion_m_rad": (-0.01, 0.0)},
    ],
)
def test_invalid_relief_proof_never_admits(change):
    estimate, proposal, evidence, limits = local_inputs()
    cert = replace(relief(estimate.local.patches[0]), **change)
    receipt = admit_local_action(
        estimate, proposal, replace(evidence, target_relief_exclusion=cert), limits, 1, generation=0
    )
    assert not receipt.admitted and receipt.reason == "invalid_target_relief_exclusion"


def refined_relief():
    from alexdoor_xas.perception.local_admission import TargetReliefBox

    estimate, _, _, _ = local_inputs()
    original = estimate.local.patches[0]
    zero = replace(original.pose_support, position_bound_m=0.0, rotation_bound_rad=0.0)
    patch = replace(
        original,
        world_pose=ObjectFrame(np.zeros(3), np.eye(3)),
        geometry_support=replace(zero, rotation_bound_rad=0.2),
        pose_support=zero,
    )
    box = TargetReliefBox(
        tuple(map(tuple, np.eye(3))), (-0.01, 0.02, 0.2), (0.01, 0.04, 2.0), 0.001
    )
    certificate = TargetReliefExclusion(
        patch.world_pose,
        ((0.0, 0.0), (1.0, 0.0), (0.0, 1.0)),
        (-0.05, -0.05),
        (0.05, 0.05),
        ((0.0, 0.03, 1.1, 0.902),),
        (0.0, 0.0),
        (box,),
        0.005,
        60000,
    )
    return patch, certificate


def test_complete_box_refinement_preserves_point_uncertainty_and_spatial_association():
    patch, certificate = refined_relief()
    assert replace(certificate, boxes=()).margin(patch) < 0
    assert certificate.require(patch) > 0
    changed = replace(
        patch, geometry_support=replace(patch.geometry_support, position_bound_m=0.2)
    )
    with pytest.raises(ValueError, match="target_relief_exclusion_exhausted"):
        certificate.require(changed)
    shifted = replace(
        patch, world_pose=replace(patch.world_pose, origin=np.array([0.0, 0.0, 0.2]))
    )
    with pytest.raises(ValueError, match="target_relief_exclusion_exhausted"):
        certificate.require(shifted)


def test_complete_box_refinement_keeps_query_and_resolution_refusals():
    patch, certificate = refined_relief()
    with pytest.raises(ValueError, match="target_relief_query_budget_exhausted"):
        replace(certificate, maximum_queries=1).require(patch)
    with pytest.raises(ValueError, match="target_relief_exclusion_exhausted"):
        replace(certificate, minimum_cell_m=2.0).require(patch)


def test_complete_box_boundary_and_model_reserve_are_strict():
    from alexdoor_xas.perception.local_admission import TargetReliefBox

    patch, certificate = refined_relief()
    zero = replace(patch.geometry_support, rotation_bound_rad=0.0)
    patch = replace(patch, geometry_support=zero)
    point = TargetReliefBox(
        tuple(map(tuple, np.eye(3))), (0.0, 0.0, 0.25), (0.0, 0.0, 0.25), 0.2
    )
    boundary = replace(
        certificate,
        spheres=((0.0, 0.0, 0.25, 0.2),),
        boxes=(point,),
    )
    assert boundary.margin(patch) == 0.0
    with pytest.raises(ValueError, match="target_relief_exclusion_exhausted"):
        boundary.require(patch)
    with pytest.raises(ValueError, match="target_relief_exclusion_exhausted"):
        replace(boundary, boxes=(replace(point, radius_m=0.20001),)).require(patch)


@pytest.mark.parametrize(
    "change",
    [
        {"boxes": ()},
        {"minimum_cell_m": 0.0},
        {"maximum_queries": 0},
        {"maximum_queries": 1.5},
    ],
)
def test_missing_refinement_or_invalid_limits_do_not_substitute_for_a_sphere(change):
    patch, certificate = refined_relief()
    with pytest.raises(ValueError, match="target_relief"):
        replace(certificate, **change).require(patch)


@pytest.mark.parametrize(
    "change",
    [
        {"axes_world": ((0.0, 0.0, 0.0),) * 3},
        {"lower_m": (np.nan, 0.0, 0.0)},
        {"lower_m": (0.0, 0.0, 3.0)},
        {"radius_m": -0.001},
    ],
)
def test_invalid_complete_box_fails_closed(change):
    patch, certificate = refined_relief()
    with pytest.raises(ValueError, match="invalid_target_relief_box"):
        replace(certificate, boxes=(replace(certificate.boxes[0], **change),)).require(patch)
