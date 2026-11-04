"""A weak fresh observation must not renew or erase a stronger bounded old one."""

from copy import deepcopy
from dataclasses import replace

import numpy as np
import pytest
from test_normal_support import scene

from alexdoor_xas.perception.contracts import (
    MAX_DYNAMIC_AGE_S,
    FieldSupport,
    validate_local_contact,
)
from alexdoor_xas.perception.point2pose.tracking import LocalSurfaceCertificate


def observed_certificate():
    tracker, candidate, sample, result = scene()
    tracker.calibration_bound = (0.0001, 0.0001)
    tracker.relative_speed_bound = (0.1, 0.15)
    candidate.support = FieldSupport(0, 0, 0, tracker.generation, 0.001, 0.001)
    candidate.ownership = "observed_rigid_material"
    tracker.select_zone(candidate.candidate_id, 0, source="policy")
    tracker.constrain_local(sample, result)
    certificate = candidate.local_certificate
    assert isinstance(certificate, LocalSurfaceCertificate)
    return tracker, candidate, sample, result, certificate


def weaker_update(tracker, candidate, sample, certificate, now=1 / 30):
    sample["time_s"] = now
    candidate.support = replace(candidate.support, acquired_s=now, available_s=now, supported_s=now)
    candidate.local_constraint = replace(certificate.constraint, position_bound_m=0.2)
    candidate.local_certificate = None
    tracker.retain_local_certificate(candidate, certificate, sample)


def test_retention_keeps_actual_camera_clock_and_all_aging_with_native_pose_unchanged():
    tracker, candidate, sample, _, certificate = observed_certificate()
    native_pose = candidate.pose.copy()
    old_state = tracker.local_state(0)
    old = old_state.patches[0]
    # This is the native camera for the next capture, not the old measurement's camera.
    candidate.camera_world = candidate.camera_world.copy()
    candidate.camera_world[2, 3] += 0.001
    weaker_update(tracker, candidate, sample, certificate)
    assert candidate.local_certificate is certificate
    assert candidate.support.supported_s == sample["time_s"]
    np.testing.assert_array_equal(candidate.pose, native_pose)
    local = tracker._constrained_candidate(candidate, candidate.local_constraint)
    np.testing.assert_array_equal(local.camera_world, certificate.camera_world)
    assert local.support is certificate.support
    patch = tracker.local_state(sample["time_s"]).patches[0]
    assert patch.pose_support.supported_s == patch.identity_support.supported_s == 0
    np.testing.assert_array_equal(patch.world_pose.origin, old.world_pose.origin)
    assert patch.pose_temporal_position_m == pytest.approx(sample["time_s"] * 0.1)
    assert patch.normal_rotation_bound_rad - old.normal_rotation_bound_rad == pytest.approx(
        sample["time_s"] * 0.15
    )
    assert patch.pose_support.position_bound_m > old.pose_support.position_bound_m
    # The finite horizon remains an independent consumer refusal.
    expired = tracker.local_state(MAX_DYNAMIC_AGE_S + 0.001).patches[0]
    assert expired.pose_support is None and expired.world_pose is None


@pytest.mark.parametrize(
    "boundary", ["expired", "generation", "selection", "reference", "identity", "depth"]
)
def test_previous_certificate_needs_current_reference_identity_depth_and_original_horizon(boundary):
    tracker, candidate, sample, _, certificate = observed_certificate()
    now = 1 / 30
    if boundary == "expired":
        now = MAX_DYNAMIC_AGE_S + 0.001
    elif boundary == "generation":
        tracker.generation += 1
    elif boundary == "selection":
        tracker.select_zone(candidate.candidate_id, now, source="policy")
    elif boundary == "reference":
        candidate.initial_world = candidate.initial_world.copy()
        candidate.initial_world[1, 3] += 0.001
    elif boundary == "identity":
        candidate.sam3_id += 1
    elif boundary == "depth":
        sample["valid_depth"][:] = False
    weaker_update(tracker, candidate, sample, certificate, now)
    assert candidate.local_certificate is not certificate
    assert candidate.local_certificate.support.supported_s == now
    assert not tracker.diagnostics["local_certificate"][candidate.candidate_id]["retained_previous"]


@pytest.mark.parametrize("failure", ["tracking_lost", "recovery", "missing", "ambiguous"])
def test_history_never_hides_unavailable_or_ambiguous_current_measurement(failure):
    tracker, candidate, sample, result, _ = observed_certificate()
    result = deepcopy(result)
    if failure == "tracking_lost":
        candidate.state = "tracking_lost"
    elif failure == "recovery":
        candidate.material_recovery_unverified = True
    elif failure == "missing":
        result["local_surfaces"] = []
    else:
        result["local_surfaces"] *= 2
    tracker.constrain_local(sample, result)
    assert candidate.local_constraint is None and candidate.local_certificate is None
    with pytest.raises(ValueError):
        validate_local_contact(tracker.local_state(0), 0)


def test_equal_fresh_certificate_is_preferred_and_reset_discards_history():
    tracker, candidate, sample, result, first = observed_certificate()
    tracker.constrain_local(sample, result)
    assert candidate.local_certificate is not first
    assert not tracker.diagnostics["local_certificate"][candidate.candidate_id]["retained_previous"]
    tracker.reset()
    assert not tracker.candidates and tracker.selection is None


@pytest.mark.parametrize("speed", [None, (None, None)])
def test_unknown_temporal_motion_never_receives_retention_credit(speed):
    tracker, candidate, sample, _, certificate = observed_certificate()
    tracker.relative_speed_bound = speed
    weaker_update(tracker, candidate, sample, certificate)
    assert candidate.local_certificate is not certificate
    assert tracker.local_state(sample["time_s"]).patches[0].normal_position_bound_m is not None


def test_stale_prior_availability_does_not_become_a_current_capture():
    tracker, candidate, sample, _, certificate = observed_certificate()
    future = replace(certificate, support=replace(certificate.support, available_s=1.0))
    weaker_update(tracker, candidate, sample, future)
    assert candidate.local_certificate is not future
    assert candidate.local_certificate.support.supported_s == sample["time_s"]
