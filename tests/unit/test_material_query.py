"""Relief cannot substitute its dominant plane for the selected observed material."""

from types import SimpleNamespace

import numpy as np
import pytest

from alexdoor_xas.perception.geometry import contact_frame, surfaces
from alexdoor_xas.perception.point2pose.tracking import PanelCandidate, PanelTracking
from alexdoor_xas.perception.scan import surface_support
from perception_helpers import camera_sensor, recipe


def setup_scene(relief, missing=False):
    sample = camera_sensor()
    mask = np.zeros((240, 240), bool)
    mask[20:221, 25:216] = True
    sample["depth_m"][:] = 2.0
    sample["depth_m"][mask] = 1.0
    if relief:
        sample["depth_m"][35:206, 40:201] = 1.0 + relief
    cue = dict(shape=mask.shape, masks=[np.packbits(mask).tobytes()], ids=[0], latency_s=0.0)
    config = dict(
        recipe().config,
        local_surface_constraint=dict(point_error_m=0.001, normal_model_bound_rad=0.001),
    )
    initial_mask = np.zeros_like(mask)
    initial_mask[20:221, 25:36] = True
    initial_cue = dict(cue, masks=[np.packbits(initial_mask).tobytes()])
    root = surfaces(initial_cue, sample, config)[0]
    world = np.eye(4)
    world[:3, :3] = contact_frame(root.normal)
    world[:3, 3] = root.points[0]
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
        sam3_id=0,
    )
    source = (root.points[::100] - sample["camera_world"][:3, 3]) @ sample["camera_world"][:3, :3]
    obj = dict(
        source_points=source,
        current_points=source.copy(),
        measured_correspondences=np.ones(len(source), bool),
        inliers=np.ones(len(source), bool),
    )
    if missing:
        sample["valid_depth"][mask & (abs(sample["depth_m"][..., 0] - 1) < 0.001)] = False
    result = dict(
        objects=[obj],
        cue=cue,
        local_surfaces=surfaces(cue, sample, config, dominant_only=True),
        local_surfaces_dominant=True,
    )
    tracking = PanelTracking(SimpleNamespace(reset=lambda: None), config)
    tracking.candidates = [candidate]
    return tracking, candidate, sample, result


@pytest.mark.parametrize("relief", [-0.04, 0.04])
def test_selected_material_survives_a_larger_current_relief_plane(relief):
    tracking, candidate, sensor, result = setup_scene(relief)
    assert tracking.root_consistent(candidate, sensor)
    assert all(abs(s.offset - 1) > 0.02 for s in result["local_surfaces"])
    native = candidate.pose.copy()
    tracking.constrain_local(sensor, result)
    assert candidate.local_constraint is not None
    local = tracking._constrained_candidate(candidate, candidate.local_constraint)
    assert tracking.root_consistent(local, sensor)
    np.testing.assert_array_equal(candidate.pose, native)
    assert tracking.diagnostics["local_surface_query"]["expanded"]
    assert tracking.diagnostics["local_surface_query_expansions"] == 1
    assert candidate.local_constraint.position_bound_m >= 0.001


def test_missing_selected_material_remains_unavailable():
    tracking, candidate, sensor, result = setup_scene(0.04, missing=True)
    tracking.constrain_local(sensor, result)
    assert candidate.local_constraint is None
    assert tracking.diagnostics["local_surface_query"]["expanded"]
    assert tracking.diagnostics["local_surface_query_expansions"] == 1


def test_nominal_plane_keeps_the_existing_query(monkeypatch):
    tracking, candidate, sensor, result = setup_scene(0)
    import alexdoor_xas.perception.point2pose.tracking as module

    monkeypatch.setattr(
        module,
        "surfaces",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("No extra query on a compatible plane")
        ),
    )
    tracking.constrain_local(sensor, result)
    assert candidate.local_constraint is not None
    assert not tracking.diagnostics["local_surface_query"]["expanded"]
