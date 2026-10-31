"""Only newly measured depth of the tracked material may condition its normal."""

from types import SimpleNamespace

import numpy as np
import pytest

from alexdoor_xas.perception.geometry import contact_frame, project, surfaces
from alexdoor_xas.perception.local_surface import observed_basis
from alexdoor_xas.perception.point2pose.tracking import PanelCandidate, PanelTracking
from alexdoor_xas.perception.scan import surface_support
from perception_helpers import camera_sensor, recipe


def scene(*, narrow_root=False, perturbation=0.0):
    sample = camera_sensor()
    mask = np.zeros((240, 240), bool)
    mask[25:216, 25:216] = True
    sample["depth_m"][mask] = 2
    config = dict(
        recipe().config,
        local_surface_constraint=dict(point_error_m=0.001, normal_model_bound_rad=0.001),
    )
    rootmask = mask.copy()
    if narrow_root:
        rootmask[:] = False
        rootmask[25:216, 35:47] = True
    cue = dict(shape=mask.shape, masks=[np.packbits(rootmask).tobytes()], ids=[3], latency_s=0.0)
    sample["depth_m"][rootmask] = 1
    root = surfaces(cue, sample, config)[0]
    world = np.eye(4)
    world[:3, :3] = contact_frame(root.normal)
    world[:3, 3] = root.points[len(root.points) // 2]
    candidate = PanelCandidate(
        "root",
        root,
        world,
        np.linalg.inv(sample["camera_world"]) @ world,
        np.eye(4),
        surface_support(root, 0, config),
        pose=world.copy(),
        state="tracked",
        camera_world=sample["camera_world"].copy(),
        sam3_id=3,
    )
    currentmask = np.zeros_like(mask)
    currentmask[25:216, 35:47] = True
    currentmask[25:216, 188:205] = True
    sample["depth_m"][mask] = 2
    sample["depth_m"][currentmask] = 1
    rows, columns = np.indices(mask.shape)
    sample["depth_m"][currentmask, 0] += perturbation * np.sin(
        rows[currentmask] * 0.2 + columns[currentmask]
    )
    cue["masks"] = [np.packbits(currentmask).tobytes()]
    current = surfaces(cue, sample, config, dominant_only=True)[0]
    uv, z = project(root.points, sample["intrinsics"], sample["camera_world"])
    uv = np.rint(uv).astype(int)
    keep = sample["valid_depth"][uv[:, 1], uv[:, 0], 0] & (
        abs(sample["depth_m"][uv[:, 1], uv[:, 0], 0] - z) <= 0.004
    )
    source = (root.points[keep] - sample["camera_world"][:3, 3]) @ sample["camera_world"][:3, :3]
    obj = dict(
        source_points=source,
        current_points=source.copy(),
        measured_correspondences=np.ones(len(source), bool),
        inliers=np.ones(len(source), bool),
    )
    tracker = PanelTracking(SimpleNamespace(reset=lambda: None), config)
    tracker.candidates = [candidate]
    return tracker, candidate, sample, dict(objects=[obj], local_surfaces=[current])


@pytest.mark.parametrize("perturbation", [0.0, 0.00015])
def test_same_material_reobservation_retains_baselines_without_changing_native_or_masks(
    perturbation,
):
    tracker, candidate, sample, result = scene(perturbation=perturbation)
    current = result["local_surfaces"][0]
    mask = current.observations[0].mask().copy()
    points = current.points.copy()
    native = candidate.pose.copy()
    local_bound = observed_basis(current, 0.001)[2]
    tracker.constrain_local(sample, result)
    assert candidate.local_constraint is not None
    assert candidate.local_constraint.normal_rotation_bound_rad < 0.02 < local_bound
    np.testing.assert_array_equal(candidate.pose, native)
    np.testing.assert_array_equal(current.points, points)
    np.testing.assert_array_equal(current.observations[0].mask(), mask)
    assert not mask[120, 120] and not mask[120, 40]
    assert tracker.root_consistent(
        tracker._constrained_candidate(candidate, candidate.local_constraint), sample
    )


def test_unrelated_coplanar_component_never_supplies_tracked_material_baselines():
    tracker, candidate, sample, result = scene(narrow_root=True)
    measured = tracker.reobserved_material_points(candidate, sample)
    uv, _ = project(measured, sample["intrinsics"], sample["camera_world"])
    uv = np.rint(uv).astype(int)
    assert ((uv[:, 0] >= 35) & (uv[:, 0] < 47)).all()
    assert observed_basis(result["local_surfaces"][0], 0.001, normal_points=measured)[2] > 0.02


def test_missing_current_material_depth_cannot_reuse_initial_xyz_or_other_components():
    tracker, candidate, sample, result = scene(narrow_root=True)
    sample["valid_depth"][:, 35:47] = False
    measured = tracker.reobserved_material_points(candidate, sample)
    assert measured.shape == (0, 3)
    tracker.constrain_local(sample, result)
    assert candidate.local_constraint is None
    assert candidate.local_constraint_reason == "invalid_observed_normal_support"


def test_reobserved_support_uses_fresh_depth_after_native_material_motion():
    tracker, candidate, sample, _ = scene()
    candidate.pose[:3, 3] += [0.03, 0, 0]
    sample["depth_m"][sample["valid_depth"]] = 1.03
    measured = tracker.reobserved_material_points(candidate, sample)
    assert len(measured) > 5
    np.testing.assert_allclose(measured[:, 0], 1.03, atol=1e-12)
    assert np.all(candidate.surface.points[:, 0] == 1.0)


@pytest.mark.parametrize("native_camera", [False, True])
def test_published_normal_primitives_retain_contact_contract_after_camera_transport(native_camera):
    from dataclasses import replace

    from scipy.spatial.transform import Rotation

    from alexdoor_xas.perception.contracts import (
        FieldSupport,
        LocalContactSelection,
        validate_local_contact,
    )

    tracker, candidate, sample, result = scene(perturbation=0.00015)
    # Native registration misses a small measured normal translation; the plane
    # constraint corrects it using the fresh depth rather than the native pose.
    current = result["local_surfaces"][0]
    current.points[:, 0] += 0.003
    current.offset = float(current.normal @ current.points.mean(0))
    sample["depth_m"][sample["depth_m"] < 1.5] += 0.003
    if native_camera:
        # The camera is a runtime float32 transform, rather than an exact
        # orthogonal identity. Transport the actual observations consistently.
        transform = np.eye(4)
        transform[:3, :3] = Rotation.from_euler("zyx", [0.37, -0.19, 0.13]).as_matrix()
        transform[:3, 3] = [0.27, -0.42, 0.31]
        transform = transform.astype(np.float32).astype(float)
        sample["camera_world"] = transform @ sample["camera_world"]
        for surface in [candidate.surface, result["local_surfaces"][0]]:
            surface.points[:] = surface.points @ transform[:3, :3].T + transform[:3, 3]
            surface.normal[:] = transform[:3, :3] @ surface.normal
            surface.offset = float(surface.normal @ surface.points.mean(0))
        candidate.pose = transform @ candidate.pose
        candidate.initial_world = transform @ candidate.initial_world
        candidate.camera_world = sample["camera_world"].copy()
    candidate.ownership = "observed_rigid_material"
    candidate.support = FieldSupport(0.0, 0.0, 0.0, 0, 0.001, 0.001)
    tracker.calibration_bound = (1.2e-5, 2.7e-5)
    tracker.relative_speed_bound = (0.1, 0.15)
    tracker.constrain_local(sample, result)
    assert candidate.local_constraint is not None
    state = tracker.local_state(0.0)
    patch = replace(state.patches[0], finger_clearance_m=(0.1, 0.1))
    state = replace(
        state,
        patches=(patch,),
        selection=LocalContactSelection("zone", candidate.candidate_id, 0.0, "policy"),
    )
    assert validate_local_contact(state, 0.0) is patch
