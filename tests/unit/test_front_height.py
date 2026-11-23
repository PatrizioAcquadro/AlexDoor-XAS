"""Finite observed height proofs retain all foreground and complete receipt volumes."""

from dataclasses import replace

import numpy as np
import pytest

from alexdoor_xas.action.spaces import A1_JOINT_DELTA, A2_EE_DELTA
from alexdoor_xas.perception.admission import require_current_admission
from alexdoor_xas.perception.contact_geometry import observed_front_extent
from alexdoor_xas.perception.local_admission import (
    TargetFrontHeightExclusion,
    TargetReliefBox,
    TargetReliefExclusion,
    admit_local_action,
)
from perception_helpers import local_inputs, measured_surface, recipe


def height_proof(patch):
    spatial = TargetReliefExclusion(
        patch.world_pose,
        ((0.0, 0.0), (1.0, 0.0), (0.0, 1.0)),
        (-0.02, 1.08),
        (0.02, 1.12),
        ((0.9, 0.045, 1.1, 0.01),),
        (0.0, 0.0),
        tangent_domain_m=((-0.1, 1.0), (0.08, 1.2)),
    )
    return TargetFrontHeightExclusion(spatial, 0.02, 0.003)


@pytest.mark.parametrize("space", [A1_JOINT_DELTA, A2_EE_DELTA])
@pytest.mark.parametrize("loaded", [False, True])
@pytest.mark.parametrize("change", ["root_exit", "normal_growth"])
def test_current_receipt_rechecks_finite_root_and_normal_for_complete_height_volume(
    space, loaded, change
):
    estimate, proposal, evidence, limits = local_inputs(space)
    patch = replace(
        estimate.local.patches[0], normal_position_bound_m=0.001, normal_rotation_bound_rad=0.001
    )
    estimate = replace(estimate, local=replace(estimate.local, patches=(patch,)))
    certificate = height_proof(patch)
    evidence = replace(evidence, target_height_exclusions=(certificate,))
    proposal = replace(proposal, loaded=loaded)
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
    changed = (
        replace(
            patch,
            world_pose=replace(patch.world_pose, origin=patch.world_pose.origin - [0, 0.02, 0]),
        )
        if change == "root_exit"
        else replace(patch, normal_position_bound_m=0.09)
    )
    current = replace(estimate, local=replace(estimate.local, patches=(changed,)))
    with pytest.raises(ValueError, match="target_front_height_exhausted"):
        require_current_admission(receipt, current, proposal.times_s[-1])


def test_height_volume_requires_entire_finite_domain_and_original_reserves():
    estimate, _, _, _ = local_inputs()
    patch = replace(
        estimate.local.patches[0], normal_position_bound_m=0.001, normal_rotation_bound_rad=0.001
    )
    certificate = height_proof(patch)
    assert certificate.require(patch) > 0
    for spatial in [
        replace(certificate.spatial, tangent_domain_m=((-0.1, 1.0), (0.05, 1.2))),
        replace(certificate.spatial, spheres=((0.9, 0.045, 1.1, 0.04),)),
    ]:
        with pytest.raises(ValueError, match="target_front_height_exhausted"):
            replace(certificate, spatial=spatial).require(patch)
    with pytest.raises(ValueError, match="target_front_height_exhausted"):
        replace(certificate, front_extent_m=0.1).require(patch)
    with pytest.raises(ValueError, match="missing_target_front_height_domain"):
        replace(certificate, spatial=replace(certificate.spatial, tangent_domain_m=None)).require(
            patch
        )
    with pytest.raises(ValueError, match="invalid_target_relief_domain"):
        replace(
            certificate, spatial=replace(certificate.spatial, tangent_domain_m=((0, 0), (0, 1)))
        ).require(patch)


def test_convex_proof_cannot_certify_its_larger_ball_or_an_understated_volume():
    estimate, _, _, _ = local_inputs()
    patch = replace(
        estimate.local.patches[0], normal_position_bound_m=0.001, normal_rotation_bound_rad=0.001
    )
    certificate = height_proof(patch)
    vertices = ((0.88, 0.04, 1.09), (0.88, 0.05, 1.11))
    # The large sphere reaches the reference front. The entire thin physical
    # convex set is behind it and within the root, with the SAME normal reserve.
    sphere = (0.88, 0.045, 1.1, 0.15)
    box = TargetReliefBox(tuple(map(tuple, np.eye(3))), vertices[0], vertices[1], 0.001)
    spatial = replace(
        certificate.spatial,
        spheres=(sphere,),
        boxes=(box,),
        minimum_cell_m=0.005,
        maximum_queries=60000,
    )
    convex = replace(certificate, spatial=spatial, vertices_world=vertices, radius_m=0.001)
    assert convex.require(patch) > 0
    with pytest.raises(ValueError, match="target_front_height_exhausted"):
        replace(convex, vertices_world=(), radius_m=0.0).require(patch)
    with pytest.raises(ValueError, match="invalid_target_front_height_volume"):
        replace(convex, spatial=replace(spatial, spheres=((0.88, 0.045, 1.1, 0.001),))).require(
            patch
        )
    with pytest.raises(ValueError, match="target_front_height_exhausted"):
        replace(
            convex, spatial=replace(spatial, tangent_domain_m=((-0.1, 1.0), (0.04, 1.2)))
        ).require(patch)


def test_height_future_growth_only_spends_remaining_interval_and_new_uncertainty():
    estimate, _, _, _ = local_inputs()
    patch = replace(
        estimate.local.patches[0], normal_position_bound_m=0.001, normal_rotation_bound_rad=0.001
    )
    certificate = height_proof(patch)
    certificate = replace(
        certificate, spatial=replace(certificate.spatial, interval_motion_m_rad=(0.001, 0.002))
    )
    assert certificate.margin(patch, remaining_fraction=0) > certificate.margin(patch)
    changed = replace(
        patch, geometry_support=replace(patch.geometry_support, position_bound_m=0.03)
    )
    with pytest.raises(ValueError, match="target_front_height_exhausted"):
        certificate.require(changed, remaining_fraction=0)


def test_observed_layers_keep_original_global_height_and_tangent_reserves_and_detached_returns():
    surface, sensor, _ = measured_surface()
    config = recipe().config
    sensor["depth_m"][110:130, 110:130, 0] = 0.94
    sensor["depth_m"][60:70, 80:90, 0] = 0.4
    views = [dict(sensor, available_s=float(sensor["time_s"]))]
    original = observed_front_extent(surface, views, config)
    assert "height_layers" not in original
    enabled = observed_front_extent(surface, views, dict(config, front_height_layers=True))
    assert {k: enabled[k] for k in original} == original
    assert enabled["rigid_connected_voxels"] < enabled["observed_voxels"]
    np.testing.assert_array_equal(
        enabled["height_layer_tangent_domain_m"], np.asarray(surface.bounds)[:, 1:]
    )
    reserve = original["front_extent_m"] - original["observed_extent_m"]
    layers = enabled["height_layers"]
    assert len(layers) > 1
    assert all(
        layer["front_extent_m"] - layer["observed_height_level_m"] == pytest.approx(reserve)
        for layer in layers
    )
    # High isolated foreground cannot disappear at the planar/handle levels.
    assert all(layer["higher_return_voxels"] > 0 for layer in layers)
    assert layers[-1]["higher_return_voxels"] < layers[0]["higher_return_voxels"]
    assert np.all(np.asarray(layers[-1]["upper_m"]) >= np.asarray(layers[-1]["lower_m"]))
    with pytest.raises(ValueError, match="invalid_front_height_layers_option"):
        observed_front_extent(surface, views, dict(config, front_height_layers=1))


def test_missing_normal_measurement_never_admits_height_layer():
    estimate, _, _, _ = local_inputs()
    patch = estimate.local.patches[0]
    with pytest.raises(ValueError, match="invalid_target_front_height_uncertainty"):
        height_proof(patch).require(patch)
