"""Numerical producer contracts; scene responses below are explicit test fixtures."""

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from alexdoor_xas.action.spaces import A1_JOINT_DELTA
from alexdoor_xas.assets.purdue import ARM_JOINTS
from alexdoor_xas.kinematics.urdf_geometry import RobotGeometry
from alexdoor_xas.perception.admission import require_current_admission
from alexdoor_xas.perception.workspace import StaticWorkspaceSupport
from alexdoor_xas.policies.pilot_runtime import PilotMotionEvidence
from perception_helpers import local_inputs


def producer(*, contact=False, unknown=False, load_refused=False):
    estimate, proposal, _, _ = local_inputs(A1_JOINT_DELTA)
    patch = replace(
        estimate.local.patches[0],
        normal_position_bound_m=0.001,
        normal_rotation_bound_rad=0.001,
    )
    estimate = replace(estimate, local=replace(estimate.local, patches=(patch,)))
    observation = SimpleNamespace(time_s=1.0, generation=0, estimate=estimate)
    tool = patch.world_pose
    model = object.__new__(RobotGeometry)
    model.shapes = [dict(link=f"body{i}", planes=np.empty((0, 4))) for i in range(2)]
    # Both roots need normal-only tightening when separated. A final leaf of
    # body1 instead lacks separation in the contact fixture, fixing that branch.
    for i, shape in enumerate(model.shapes):
        center = np.array([1.0 if contact and i == 1 else 0.95, 0.1 * i, 1.1])
        shape.update(lower=center - 0.02, upper=center + 0.02)
    matrices = {shape["link"]: np.eye(4) for shape in model.shapes}
    from alexdoor_xas.assets.purdue import PUSH_PARENT

    matrices[PUSH_PARENT] = np.eye(4)
    model.parents = {}
    model.transforms = lambda *_: matrices
    model.velocity_sweep = lambda *_: (
        lambda cell: (
            (cell.lower + cell.upper) / 2,
            np.linalg.norm(cell.upper - cell.lower) / 2,
        )
    )
    queried = []

    def query_static(centers, radii, **kwargs):
        queried.append((centers.copy(), radii.copy()))
        # Unknown body0 must still refuse even after body1 fixed contact mode.
        margin = None if unknown and centers[0, 1] < 0.05 else 0.1
        return StaticWorkspaceSupport(margin, margin is None, False, ())

    motion_bounds = []

    def check_load_motion(lower, **kwargs):
        motion_bounds.append((lower, kwargs))
        if load_refused:
            raise ValueError("outside_fixture_load_position_support")

    cover = SimpleNamespace(
        support=patch.pose_support,
        normal_range_m=(0, 0),
        query=lambda *_: SimpleNamespace(finger_clearance_m=(0.2, 0.2)),
    )
    candidate = SimpleNamespace(
        candidate_id=patch.patch_id,
        material_cover=cover,
        local_constraint=None,
        initial_world=np.eye(4),
        local_surface=None,
        surface=None,
    )
    tracking = SimpleNamespace(candidates=[candidate], point_budget_query=lambda *_: None)
    env = SimpleNamespace(
        capture=SimpleNamespace(sample=SimpleNamespace(time_s=1.0)),
        robot=SimpleNamespace(
            joint_names=ARM_JOINTS, data=SimpleNamespace(joint_pos=np.zeros((1, 7)))
        ),
        load_samples=[SimpleNamespace(dq=np.zeros(7))],
    )
    evidence = object.__new__(PilotMotionEvidence)
    evidence.io = SimpleNamespace(
        env=env, dt=1 / 60, calibration={"root_world": np.eye(4)}, preview=lambda _: tool
    )
    evidence.provider = SimpleNamespace(tracking=tracking, config={"plane_tolerance_m": 0.001})
    evidence.space = A1_JOINT_DELTA
    evidence.model = model
    evidence.cells = model.cells()
    evidence.self_clearance = SimpleNamespace(query=lambda *_args, **_kwargs: (0.1, None))
    evidence.workspace = SimpleNamespace(query_static=query_static)
    evidence.background_mode = False
    evidence.stop = evidence.acceleration = np.zeros(7)
    evidence.shape_stop = np.full(2, 0.001)
    evidence.stop_boxes = None
    evidence.faces = (np.array([[0, -0.01, 0]]), np.array([[0, 0.01, 0]]))
    evidence.load_calibration = SimpleNamespace(
        require_position_motion_support=check_load_motion,
        q_min=-np.ones(7),
        q_max=np.ones(7),
        position_tolerance_rad=0.01,
    )
    evidence.config = dict(
        max_joint_speed_rad_s=1.0,
        held_joint_travel={},
        closed_jaw_bound_m=0.001,
        robot_model_bound_m=0.001,
        relative_speed_bound_m_s=0.01,
        material_angular_speed_rad_s=0.02,
        camera_bound=(0.001, 0.001),
        normal_model_bound_m=0.001,
        maximum_cover_queries=1000,
        minimum_cell_m=0.01,
        stop_duration_s=0.1,
        stop_force_domain_n=10.0,
    )
    return evidence, observation, tool, proposal.action, queried, motion_bounds


@pytest.mark.parametrize("contact", [False, True])
def test_complete_parent_scene_proofs_and_original_receipt_branch(contact):
    evidence, observation, tool, row, queried, motion_bounds = producer(contact=contact)
    diagnostics = {}
    result = evidence.produce(observation, tool, row, 1, diagnostics)
    assert result.admitted and result.proposal.loaded is contact
    require_current_admission(result, observation.estimate, observation.time_s + 1 / 240)
    assert diagnostics["robot_cover_queries"] == len(queried)
    assert diagnostics["sweep_margin_m"] > 0
    assert len(motion_bounds) == 1 and motion_bounds[0][1]["force_upper_n"] == 10
    body0 = [cell for cell in evidence.cells if cell.shape == 0]
    if contact:
        # The complete body0 root was proved, without expanding it just to
        # improve an unloaded margin that this proposal can no longer consume.
        assert len(body0) == 1 and body0[0].path == ()
        assert result.local_evidence.unloaded_separation_margin_m is None
    else:
        assert len(body0) > 1
        assert result.local_evidence.unloaded_separation_margin_m > 0


def test_contact_branch_never_skips_unknown_complete_robot_volume():
    evidence, observation, tool, row, queried, _ = producer(contact=True, unknown=True)
    with pytest.raises(ValueError, match="unsupported_actual_robot_sweep_and_stop"):
        evidence.produce(observation, tool, row, 1, {})
    assert any(centers[0, 1] < 0.05 for centers, _ in queried)


def test_contact_branch_never_skips_position_and_full_stop_load_support():
    evidence, observation, tool, row, _, motion_bounds = producer(contact=True, load_refused=True)
    with pytest.raises(ValueError, match="outside_fixture_load_position_support"):
        evidence.produce(observation, tool, row, 1, {})
    assert len(motion_bounds) == 1


@pytest.mark.parametrize("offset_m", [-1e-5, 0.0, 1e-5])
def test_strict_zero_separation_boundary_keeps_unloaded_refinement(offset_m):
    evidence, observation, tool, row, _, _ = producer()
    patch = replace(
        observation.estimate.local.patches[0],
        normal_position_bound_m=0.0,
        normal_rotation_bound_rad=0.0,
    )
    observation.estimate = replace(
        observation.estimate, local=replace(observation.estimate.local, patches=(patch,))
    )
    evidence.config.update(
        robot_model_bound_m=0.0,
        normal_model_bound_m=0.0,
        relative_speed_bound_m_s=0.0,
        material_angular_speed_rad_s=0.0,
    )
    evidence.shape_stop[:] = 0.0
    first = evidence.model.shapes[1]
    first["lower"] = first["upper"] = np.array([1.0 + offset_m, 0.1, 1.1])
    other = evidence.model.shapes[0]
    other["lower"], other["upper"] = (np.array([0.96, 0, 1.1]) + side * 0.02 for side in (-1, 1))
    evidence.cells = evidence.model.cells()
    result = evidence.produce(observation, tool, row, 1, {})
    assert result.admitted and result.proposal.loaded is (offset_m >= 0)
    body0 = [cell for cell in evidence.cells if cell.shape == 0]
    assert (len(body0) == 1) is (offset_m >= 0)


def test_repeated_contact_proofs_do_not_expand_unneeded_normal_cells():
    evidence, observation, tool, row, queried, _ = producer(contact=True)
    counts = []
    for _ in range(3):
        start = len(queried)
        result = evidence.produce(observation, tool, row, 1, {})
        assert result.admitted and result.proposal.loaded
        counts.append(len(queried) - start)
        assert len([cell for cell in evidence.cells if cell.shape == 0]) == 1
    assert counts[1] == counts[2] < counts[0]
    # A new measured separation must restore the unloaded branch and normal
    # refinement, despite the previous command's contact-prioritized traversal.
    shape = evidence.model.shapes[1]
    shift = np.array([-0.2, 0.0, 0.0])
    evidence.cells = tuple(
        replace(cell, lower=cell.lower + shift, upper=cell.upper + shift)
        if cell.shape == 1
        else cell
        for cell in evidence.cells
    )
    shape["lower"], shape["upper"] = shape["lower"] + shift, shape["upper"] + shift
    result = evidence.produce(observation, tool, row, 1, {})
    assert result.admitted and not result.proposal.loaded
    assert len([cell for cell in evidence.cells if cell.shape == 0]) > 1


def test_old_fine_partition_can_coarsen_without_losing_physical_volume():
    evidence, observation, tool, row, queried, _ = producer(contact=True)
    cells = evidence.model.cells()
    for _ in range(2):
        cells = tuple(child for cell in cells for child in evidence.model.subdivide(cell))
    evidence.cells = cells
    counts = []
    for _ in range(2):
        start = len(queried)
        result = evidence.produce(observation, tool, row, 1, {})
        assert result.admitted and result.proposal.loaded
        require_current_admission(result, observation.estimate, observation.time_s + 1 / 240)
        counts.append(len(queried) - start)
        # Independent axis-aligned fixture: disjoint interiors and exactly the
        # same complete root volume, rather than merely the old leaf centers.
        for index, shape in enumerate(evidence.model.shapes):
            parts = [cell for cell in evidence.cells if cell.shape == index]
            lower = np.stack([cell.lower for cell in parts])
            upper = np.stack([cell.upper for cell in parts])
            assert np.all(lower >= shape["lower"]) and np.all(upper <= shape["upper"])
            assert np.isclose(
                np.prod(upper - lower, axis=1).sum(),
                np.prod(shape["upper"] - shape["lower"]),
                rtol=2e-14,
                atol=1e-18,
            )
            for i in range(len(parts)):
                overlap = np.minimum(upper[i], upper[i + 1 :]) - np.maximum(
                    lower[i], lower[i + 1 :]
                )
                assert not np.any(np.all(overlap > 1e-15, axis=1))
    assert counts[1] < counts[0]


@pytest.mark.parametrize("missing", ["scene", "material", "load"])
def test_coarsened_next_traversal_requires_fresh_complete_evidence(missing):
    evidence, observation, tool, row, _, _ = producer(contact=True)
    cells = evidence.model.cells()
    for _ in range(2):
        cells = tuple(child for cell in cells for child in evidence.model.subdivide(cell))
    evidence.cells = cells
    assert evidence.produce(observation, tool, row, 1, {}).admitted
    if missing == "scene":
        previous = evidence.workspace.query_static

        def unknown(centers, radii, **kwargs):
            if centers[0, 1] < 0.05:
                return StaticWorkspaceSupport(None, True, False, ())
            return previous(centers, radii, **kwargs)

        evidence.workspace.query_static = unknown
        with pytest.raises(ValueError, match="unsupported_actual_robot_sweep_and_stop"):
            evidence.produce(observation, tool, row, 1, {})
    elif missing == "material":
        cover = evidence.provider.tracking.candidates[0].material_cover
        cover.query = lambda *_: SimpleNamespace(finger_clearance_m=(None, None))
        result = evidence.produce(observation, tool, row, 1, {})
        assert not result.admitted and result.reason == "missing_relevant_clearance"
    else:

        def unsupported(*_args, **_kwargs):
            raise ValueError("outside_fixture_load_position_support")

        evidence.load_calibration.require_position_motion_support = unsupported
        with pytest.raises(ValueError, match="outside_fixture_load_position_support"):
            evidence.produce(observation, tool, row, 1, {})
