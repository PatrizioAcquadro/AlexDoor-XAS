"""Prepared geometry and the two-cycle expert admission contract."""

import json
from itertools import product

import numpy as np
import pytest
from scipy.spatial import ConvexHull

from alexdoor_xas.qualification.door_geometry import PreparedDoor
from alexdoor_xas.qualification.expert import (
    continue_pair,
    publish_expert,
    qualify_pair,
    trial_summary,
)
from alexdoor_xas.qualification.synthetic_probe import SustainedAngle


@pytest.fixture
def trial():
    return dict(
        case="door",
        passed=True,
        released=True,
        angle_deg=50.0,
        hold_angle_deg=49.9,
        stop_reason="safety_stop",
        safety_detail="tracking_margin",
        joint_margin=0.01,
        peak_force_n=2.0,
        clearance_m=0.0,
    )


def test_pair_requires_two_complete_consistent_cycles(trial):
    assert qualify_pair([trial])["status"] == "unresolved"
    assert qualify_pair([trial] * 3)["status"] == "unresolved"
    assert qualify_pair([trial, {**trial, "angle_deg": 52}])["theta_expert_d"] == 50
    for replacement in (
        {"angle_deg": 52.01},
        {"stop_reason": "timeout"},
        {"stop_reason": "solver_tracking_stall"},
        {"safety_detail": "high_normal_force"},
        {"released": False},
        {"hold_angle_deg": None},
        {"hold_angle_deg": float("nan")},
        {"passed": False},
    ):
        assert qualify_pair([trial, {**trial, **replacement}])["status"] == "unresolved"


def test_admission_is_inclusive_and_uses_lower_maximum(trial):
    low = {**trial, "angle_deg": 45.0}
    result = qualify_pair([low, {**low, "angle_deg": 46.0}])
    assert result["status"] == "qualified" and result["theta_expert_d"] == 45.0
    low["angle_deg"] = 44.99
    assert qualify_pair([low, low])["status"] == "out_of_domain"


@pytest.mark.parametrize(
    "change", [None, "input", "setup", "device", "completed", "diagnostic", "summary", "second"]
)
def test_process_continuation_cannot_mix_inputs_or_retry_a_second_trial(tmp_path, trial, change):
    folder, output = tmp_path / "door", tmp_path / "evidence"
    folder.mkdir()
    (output / "repeat-1").mkdir(parents=True)
    config = tmp_path / "setup.json"
    config.write_text("{}")
    (output / "setup.json").write_text("{}")
    for name in ("prepared.json", "recipe.json", "candidate.json"):
        (folder / name).write_text("{}")
        (output / name).write_text("{}")
    trial = {**trial, "visibility": None}
    report = dict(
        asset_id="door",
        device="cuda:0",
        diagnostic_only=False,
        status="unresolved",
        reason="incomplete_execution",
        trials=[trial_summary(trial, 1)],
    )
    (output / "repeat-1/result.json").write_text(json.dumps(trial))
    if change == "input":
        (folder / "recipe.json").write_text('{"changed": true}')
    elif change == "setup":
        config.write_text('{"changed": true}')
    elif change == "device":
        report["device"] = "cuda:1"
    elif change == "completed":
        report["status"] = "qualified"
    elif change == "diagnostic":
        report["diagnostic_only"] = True
    elif change == "summary":
        report["trials"][0]["angle_deg"] += 1
    elif change == "second":
        (output / "repeat-2").mkdir()
    (output / "report.json").write_text(json.dumps(report))
    if change is None:
        assert continue_pair(output, folder, config, "cuda:0") == (report, [trial])
    else:
        with pytest.raises(ValueError):
            continue_pair(output, folder, config, "cuda:0")


def test_sustained_maximum_survives_lower_hold_and_release():
    window = SustainedAngle(0.5)
    for tick in range(11):
        window.update(tick / 10, 0.9 if tick < 6 else 0.8, True)
    for tick in range(11, 20):
        window.update(tick / 10, 0.8, True)
    window.update(2.0, 1.5, False)
    assert window.maximum == 0.9


def make_door(tmp_path, hand):
    folder = tmp_path / hand
    (folder / "prepared").mkdir(parents=True)
    (folder / "prepared/door.usda").touch()
    sign = 1 if hand == "left" else -1
    record = dict(
        handedness=hand,
        dimensions_m=dict(width_m=0.9, height_m=2, thickness_m=0.06),
        hinge_m=[0.11, sign * 0.46, 0],
        panel_center_m=[0.07, 0, 1.03],
        mechanical_limit_deg=74.5,
        usd="prepared/door.usda",
    )
    door = PreparedDoor(folder, record, {"components": {"Panel": [0]}})
    points = np.array(list(product([0.04, 0.10], [-0.45, 0.45], [0.03, 2.03])))
    door._leaf_planes = [ConvexHull(points).equations]
    door.shapes = {"Panel": [points], "Frame": [points + [0.1, 0.5, 0]], "Handle": []}
    return door


@pytest.mark.parametrize("hand", ["left", "right"])
def test_prepared_contact_uses_surface_and_actual_hinge(tmp_path, hand):
    door = make_door(tmp_path, hand)
    closed, _ = door.contact_pose(0, 0.4, 1)
    assert closed[0] == pytest.approx(0.04)
    assert closed[1] == pytest.approx(door.hinge[1] - door.sign * 0.36)
    opened, rotation = door.contact_pose(0.7, 0.4, 1)
    np.testing.assert_allclose((opened - door.hinge) @ rotation + door.hinge, closed)
    assert door.mechanical_stop == pytest.approx(np.deg2rad(74.5))
    assert door.footprint_inside(np.array([opened]), 0.7)
    assert not door.footprint_inside(np.array([[0, 4, 1]]), 0)
    with pytest.raises(ValueError, match="no collidable"):
        door.front_x(0.5, 1)
    assert all(name != "handle" for name, _ in door.collision_bounds(0.7))


def test_publication_preserves_preparation_and_detects_concurrent_edit(tmp_path):
    original = {
        "status": "ready_for_5.1",
        "expert_qualification": "not_run",
        "checks": {"physics": "pass"},
        "usd": "prepared/door.usda",
    }
    path = tmp_path / "prepared.json"
    path.write_text(json.dumps(original))
    report = dict(status="unresolved", theta_expert_d=None, reason="timeout", trials=[])
    publish_expert(tmp_path, original, report, tmp_path / "evidence")
    updated = json.loads(path.read_text())
    assert updated["checks"] == original["checks"]
    assert updated["usd"] == original["usd"]
    assert updated["expert_qualification"]["status"] == "unresolved"
    with pytest.raises(RuntimeError, match="changed"):
        publish_expert(tmp_path, original, report, tmp_path / "other")


@pytest.mark.parametrize("hand", ["left", "right"])
def test_sloped_surface_orients_tool_without_moving_prescribed_point(tmp_path, hand):
    door = make_door(tmp_path, hand)
    points = door.shapes["Panel"][0].copy()
    points[:, 0] += 0.15 * points[:, 1] + 0.02 * points[:, 2]
    door._leaf_planes = [ConvexHull(points).equations]
    closed, frame = door.contact_pose(0, 0.4, 1)
    assert closed[1] == pytest.approx(door.hinge[1] - door.sign * 0.36)
    assert closed[2] == 1
    assert closed[0] == pytest.approx(0.04 + 0.15 * closed[1] + 0.02)
    normal = np.array([1.0, -0.15, -0.02])
    normal /= np.linalg.norm(normal)
    np.testing.assert_allclose(frame[:, 0], normal)
    np.testing.assert_allclose(frame.T @ frame, np.eye(3), atol=1e-12)
    assert np.linalg.det(frame) == pytest.approx(1)
    opened, rotated = door.contact_pose(0.7, 0.4, 1, 0.003)
    panel = door.rotation(0.7)
    np.testing.assert_allclose(opened, door.hinge + panel @ (closed + 0.003 * normal - door.hinge))
    np.testing.assert_allclose(rotated, panel @ frame)
    face = np.array(list(product([0], [-0.02, 0.02], [-0.02, 0.02]))) @ frame.T + closed
    assert door.footprint_obstructions([face], 0.001, normal) == []
    strip = np.array(list(product([-0.02, 0], [-0.002, 0.002], [-0.01, 0.01])))
    door._leaf_planes.append(ConvexHull(strip @ frame.T + closed).equations)
    witnesses = door.footprint_obstructions([face], 0.01, normal)
    assert len(witnesses) == 1
    assert witnesses[0]["protrusion_m"] == pytest.approx(0.02)


@pytest.mark.parametrize("support", ["face", "edge", "point"])
def test_footprint_rejects_raised_strip_between_pad_corners(tmp_path, support):
    door = make_door(tmp_path, "right")
    face = np.array(list(product([0.04], [-0.02, 0.02], [0.98, 1.02])))
    if support == "edge":
        face = np.array([[0.04, -0.02, 1], [0.04, 0.02, 1]])
    elif support == "point":
        face = np.array([[0.04, 0, 1]])
    assert door.footprint_obstructions([face], 0.01) == []
    # Every corner still sees the flat base; the bar crosses the pad interior.
    strip = np.array(list(product([0.015, 0.04], [-0.002, 0.002], [0.99, 1.01])))
    door._leaf_planes.append(ConvexHull(strip).equations)
    witnesses = door.footprint_obstructions([face], 0.01)
    assert len(witnesses) == 1
    assert witnesses[0]["protrusion_m"] == pytest.approx(0.025)
    assert door.footprint_obstructions([face], 0.025) == []
    assert door.footprint_obstructions([face + [0, 0.1, 0]], 0.01) == []
