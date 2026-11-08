"""Observed geometry completes orientation without inventing native correspondence."""

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from alexdoor_xas.perception.contracts import FieldSupport
from alexdoor_xas.perception.geometry import surfaces
from alexdoor_xas.perception.observed_articulation import StaticReference
from alexdoor_xas.perception.point2pose.tracking import (
    PanelCandidate,
    PanelTracking,
    metric_registration_bound,
)
from perception_helpers import camera_sensor, recipe


def observed_case(noise=0.0, motion=0.01):
    initial, current = camera_sensor(0, 0), camera_sensor(1, 1)
    mask = np.zeros((240, 240), bool)
    mask[75:160, 105:129] = True
    initial["depth_m"][mask] = 1.0
    current["depth_m"][mask] = 1.0 + motion
    cue = dict(shape=mask.shape, masks=[np.packbits(mask).tobytes()], ids=[3], latency_s=0.0)
    config = dict(
        recipe().config,
        local_surface_constraint=dict(point_error_m=0.001, normal_model_bound_rad=0.001),
    )
    old, new = [surfaces(cue, sample, config)[0] for sample in (initial, current)]
    anchor = np.eye(4)
    anchor[:3, 3] = old.points.mean(0)
    support = FieldSupport(0, 0, 0, 0, 0.004, 0.02)
    candidate = PanelCandidate(
        "pane",
        old,
        anchor,
        np.linalg.inv(initial["camera_world"]) @ anchor,
        np.eye(4),
        support,
        static_reference=StaticReference.observed(anchor, support),
        sam3_id=3,
        p2p_object_id=0,
    )
    tracker = PanelTracking(
        SimpleNamespace(reset=lambda: None),
        config,
        calibration_bound=(0.00001, 0.00001),
        relative_speed_bound=(0.1, 0.15),
    )
    tracker.candidates = [candidate]
    # Nine measured correspondences on one vertical line cannot estimate rotation alone.
    source = np.c_[np.zeros(9), np.linspace(-0.1, 0.1, 9), np.ones(9)]
    target = source + [0, 0, motion]
    target[:, 0] += noise * np.sin(np.arange(9))
    native = np.eye(4)
    native[:3, :3] = Rotation.from_euler("z", 0.02).as_matrix()
    native[:3, 3] = [0, 0, motion + 0.001]
    obj = dict(
        object_id=0,
        source_points=source,
        current_points=target,
        measured_correspondences=np.ones(9, bool),
        inliers=np.ones(9, bool),
        camera_from_map=native,
        lost=False,
    )
    result = dict(objects=[obj], local_surfaces=[new], latency_s=0.0, capture_s=1.0)
    return tracker, candidate, current, result


@pytest.mark.parametrize("noise", [0.0, 0.0002])
@pytest.mark.parametrize("motion", [0.0, 0.01])
@pytest.mark.parametrize("pairs", [5, 9])
def test_geometry_bounds_unchanged_native_pose_with_collinear_pairs(noise, motion, pairs):
    tracker, candidate, current, result = observed_case(noise, motion)
    obj = result["objects"][0]
    native = obj["camera_from_map"].copy()
    obj["measured_correspondences"][pairs:] = False
    assert metric_registration_bound(
        obj["source_points"], obj["current_points"], native, 0.004
    ) == (None, None)
    tracker.consume(current, result, 1.0)
    assert candidate.state == "tracked"
    detail = tracker.diagnostics["support_by_candidate"][candidate.candidate_id]
    assert detail["registration_bound_source"] == "native_pairs_and_observed_surface"
    origin = candidate.registration_origin_map
    truth = np.eye(4)
    truth[:3, 3] = [0, 0, motion]
    error = np.linalg.norm(((native - truth) @ np.r_[origin, 1])[:3])
    assert error <= candidate.registration_position_m
    assert 0.02 <= candidate.registration_rotation_rad
    native_world = candidate.pose.copy()
    tracker.constrain_local(current, result)
    assert candidate.local_constraint is not None
    corrected = tracker._constrained_candidate(candidate, candidate.local_constraint)
    expected = candidate.initial_world.copy()
    expected[0, 3] += motion
    assert np.linalg.norm(corrected.pose[:3, 3] - expected[:3, 3]) <= noise + 1e-10
    np.testing.assert_array_equal(obj["camera_from_map"], native)
    np.testing.assert_array_equal(candidate.pose, native_world)
    # Only independently resolved normal motion establishes the existing moving-material claim.
    assert candidate.ownership == (
        "observed_moving_rigid_candidate" if motion else "ambiguous_leaf_candidate"
    )
    assert tracker.selection is None and not tracker.diagnostics["loaded_contact_admitted"]


@pytest.mark.parametrize(
    "missing",
    [
        "edge",
        "identity",
        "synchronized_surface",
        "depth",
        "native_tracking",
        "semantic_tracking",
        "pairs",
        "recovery",
        "metric_recipe",
        "surface_ambiguity",
        "finite_native_pose",
        "measured_flag",
    ],
)
def test_geometry_cannot_replace_missing_identity_or_metric_evidence(missing):
    tracker, candidate, current, result = observed_case()
    obj, surface = result["objects"][0], result["local_surfaces"][0]
    if missing == "edge":
        surface.edge_points = {}
    elif missing == "identity":
        surface.observations = (replace(surface.observations[0], semantic_id=4),)
    elif missing == "synchronized_surface":
        surface.observations = (replace(surface.observations[0], frame=0),)
    elif missing == "depth":
        current["depth_m"] += 0.03
    elif missing == "native_tracking":
        obj["lost"] = True
    elif missing == "semantic_tracking":
        result["semantic_absent"] = True
    elif missing == "pairs":
        obj["measured_correspondences"][4:] = False
    elif missing == "recovery":
        candidate.material_recovery_unverified = True
    elif missing == "surface_ambiguity":
        result["local_surfaces"] = [surface, surface]
    elif missing == "measured_flag":
        obj.pop("measured_correspondences")
    elif missing == "finite_native_pose":
        obj["camera_from_map"][0, 0] = np.nan
    else:
        tracker.config.pop("local_surface_constraint")
    tracker.consume(current, result, 1.0)
    assert candidate.state == "tracking_lost"
    assert candidate.support is None
    assert not tracker.diagnostics["loaded_contact_admitted"]
