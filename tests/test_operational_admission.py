"""Essential 6.0A numerical contracts; no model, simulator or physical qualification."""

from dataclasses import replace

import numpy as np
import pytest

from alexdoor_xas.action.b1 import relative_pose
from alexdoor_xas.action.frames import ObjectFrame
from alexdoor_xas.action.spaces import A2_EE_DELTA
from alexdoor_xas.assets.purdue import ARM_JOINTS
from alexdoor_xas.perception.admission import (
    ActionLimits,
    ActionProposal,
    ContactCandidate,
    HypothesisActionSupport,
    admit_action,
    admit_load,
    require_current_admission,
    rotational_travel,
    select_contact,
)
from alexdoor_xas.perception.contracts import (
    OPERATIONAL_V1,
    ContactSelection,
    DoorEstimate,
    FieldSupport,
    HingeHypothesis,
    OperationalState,
    RobotFeedback,
    geometric_admission,
    validate_contact_transition,
)


def operational_estimate(time=0.0, generation=0, count=1, bound=0.001):
    """Declared numerical support; these values are not a measured robot recipe."""
    support = FieldSupport(time, time, time, generation, bound, bound)
    hinge = ObjectFrame(np.zeros(3), np.eye(3))
    local = ObjectFrame(np.array([0.0, 0.3, 1.0]), np.eye(3))
    hypotheses = tuple(HingeHypothesis(
        f"h{i}", ObjectFrame(np.array([0.0, i * 0.3, 0.0]), np.eye(3)), 0.0, support
    ) for i in range(count))
    contact = ContactSelection("selection-1", "patch-1", time, local, local, support, support)
    state = OperationalState(
        "observed-leaf-1", generation, "tracking" if count == 1 else "ambiguous",
        support, hypotheses, "h0", support, support, contact
    )
    return DoorEstimate(time, False, "legacy_incomplete", frame=hinge,
                        panel_rotation=np.eye(3), signed_angle=0.0,
                        contact_position=local.origin, contact_rotation=local.rot,
                        operational=state)


def action_inputs(estimate):
    contact = estimate.operational.contact
    poses = (contact.world_pose, ObjectFrame(contact.world_pose.origin + [0.001, 0, 0], np.eye(3)))
    now = estimate.timestamp_s
    proposal = ActionProposal("diagnostic", A2_EE_DELTA, np.array([0.001, 0, 0, 0, 0, 0]),
                              np.array([now, now + 0.1]), poses, contact.selection_id, True)
    support = estimate.operational.identity_support
    evidence = tuple(HypothesisActionSupport(
        hinge.hypothesis_id,
        relative_pose(contact.world_pose, ObjectFrame(hinge.frame.origin, estimate.panel_rotation)),
        poses, support, (0.1, 0.1), 0.1, False, 0.001, 0.001, 1.0
    ) for hinge in estimate.operational.hypotheses)
    limits = ActionLimits(0.2, 0.01, 0.1, 0.1, 0.001, 0.001, 0.01, 0.001)
    return proposal, evidence, limits


def decide(estimate, proposal=None, evidence=None, limits=None, now=None):
    defaults = action_inputs(estimate)
    return admit_action(estimate, proposal or defaults[0], evidence or defaults[1],
                        limits or defaults[2], estimate.timestamp_s if now is None else now,
                        generation=estimate.operational.generation)


def test_operational_missing_dimensions_does_not_change_legacy_validity():
    estimate = operational_estimate()
    assert not estimate.valid and not geometric_admission(estimate, 0).qualified
    assert geometric_admission(estimate, 0, profile=OPERATIONAL_V1).qualified
    proposal, evidence, limits = action_inputs(estimate)
    assert decide(estimate).admitted
    request = replace(proposal, required_dimensions=(1,))
    assert not decide(estimate, request, evidence, limits).admitted


def test_ambiguous_and_large_uncertainty_are_provisional_not_qualified():
    estimate = operational_estimate(count=2, bound=0.02)
    assert not geometric_admission(estimate, 0, profile=OPERATIONAL_V1).qualified
    proposal, evidence, limits = action_inputs(estimate)
    evidence = tuple(replace(e, finger_clearance_m=(1.0, 1.0), collision_clearance_m=1.0)
                     for e in evidence)
    result = decide(estimate, proposal, evidence, limits)
    assert result.admitted and result.provisional and len(result.margins_m) == 2
    assert not decide(estimate, replace(proposal, source="policy"), evidence, limits).admitted
    assert not decide(estimate, proposal, evidence[:1], limits).admitted


@pytest.mark.parametrize("change", [
    dict(finger_clearance_m=(0.001, 0.1)), dict(collision_clearance_m=0.001),
    dict(unknown_in_envelope=True), dict(lever_arm_m=None),
    dict(response_position_bound_m=None), dict(finger_clearance_m=(0.1, None)),
])
def test_necessary_action_support_cannot_be_invented(change):
    estimate = operational_estimate()
    proposal, evidence, limits = action_inputs(estimate)
    assert not decide(estimate, proposal, (replace(evidence[0], **change),), limits).admitted


@pytest.mark.parametrize("change", [
    dict(stop_travel_m=0.2), dict(robot_position_bound_m=None),
    dict(relative_speed_bound_m_s=10), dict(duration_s=0.01),
    dict(displacement_m=0.0001), dict(speed_m_s=0.001), dict(angular_speed_rad_s=None),
])
def test_latency_stop_and_explicit_limits_consume_margin(change):
    estimate = operational_estimate()
    proposal, evidence, limits = action_inputs(estimate)
    assert not decide(estimate, proposal, evidence, replace(limits, **change)).admitted
    assert rotational_travel(1, np.pi * 2) == pytest.approx(2)


def test_every_hypothesis_uses_same_patch_and_world_command():
    estimate = operational_estimate(count=2)
    proposal, evidence, limits = action_inputs(estimate)
    assert decide(estimate).admitted
    wrong_local = replace(evidence[1], contact_local=evidence[0].contact_local)
    assert decide(estimate, proposal, (evidence[0], wrong_local), limits).reason == (
        "different_physical_contact"
    )
    wrong_path = replace(evidence[1], world_poses=(proposal.world_poses[0], ObjectFrame(
        np.ones(3), np.eye(3)
    )))
    assert decide(estimate, proposal, (evidence[0], wrong_path), limits).reason == (
        "different_world_trajectory"
    )


def test_static_support_lives_but_prediction_and_generation_do_not_refresh_dynamic_fields():
    estimate = operational_estimate(time=10)
    state = estimate.operational
    static = FieldSupport(0, 0, 0, 0, 0.001, 0.001)
    state = replace(state, hypotheses=(replace(state.hypotheses[0], support=static),),
                    contact=replace(state.contact, local_support=static))
    estimate = replace(estimate, timestamp_s=0, operational=state)
    assert geometric_admission(estimate, 10, profile=OPERATIONAL_V1).qualified
    predicted = replace(estimate, timestamp_s=10.2)
    assert not geometric_admission(predicted, 10.2, profile=OPERATIONAL_V1).qualified
    assert not geometric_admission(estimate, 10, profile=OPERATIONAL_V1, generation=1).qualified
    late = replace(state.angle_support, available_s=10.1)
    delayed = replace(estimate, operational=replace(state, angle_support=late))
    assert not geometric_admission(delayed,
                                   10, profile=OPERATIONAL_V1).qualified


def test_admission_snapshots_command_and_rejects_changed_identity_reference_or_uncertainty():
    estimate = operational_estimate()
    proposal, evidence, limits = action_inputs(estimate)
    receipt = decide(estimate, proposal, evidence, limits)
    proposal.action[0] = 99
    assert receipt.proposal.action[0] == 0.001
    require_current_admission(receipt, estimate, 0)
    state = estimate.operational
    for changed in (replace(state, leaf_id="other"), replace(state, reference_id="other"),
                    replace(state, contact=replace(state.contact, selection_id="new")),
                    replace(state, angle_support=replace(
                        state.angle_support, rotation_bound_rad=0.1
                    ))):
        with pytest.raises(ValueError):
            require_current_admission(receipt, replace(estimate, operational=changed), 0)


def test_diagnostic_contact_order_does_not_depend_on_unknown_total_width():
    contact = operational_estimate().operational.contact
    a = ContactCandidate(contact, True, (0.02, 0.03), None)
    b = replace(a, selection=replace(contact, selection_id="selection-2"),
                finger_clearance_m=(0.03, 0.03), hinge_distance_m=0.4)
    assert select_contact([a, b], hinge_resolved=False) == b.selection
    tied = replace(b, selection=replace(contact, selection_id="selection-0"))
    assert select_contact([b, tied], hinge_resolved=True) == tied.selection
    with pytest.raises(ValueError):
        select_contact([replace(a, reachable=False)], hinge_resolved=False)


def test_contact_transition_and_evaluator_reference_are_explicit_and_separate():
    from alexdoor_xas.perception.evaluation import MaterialContactReference

    contact = operational_estimate().operational.contact
    validate_contact_transition(None, contact)
    validate_contact_transition(contact, contact)
    changed = replace(contact, selection_id="new", selected_s=0.1)
    with pytest.raises(ValueError, match="transition"):
        validate_contact_transition(contact, changed)
    validate_contact_transition(contact, replace(changed, predecessor_id=contact.selection_id))
    reference = MaterialContactReference(contact.selection_id, contact.selected_s,
                                         contact.local_pose, (np.zeros((3, 3)),) * 2)
    reference.require_selection(contact)
    with pytest.raises(ValueError, match="correspondence"):
        reference.require_selection(changed)
    assert "leaf_local_pose" not in vars(contact)


def load_inputs():
    feedback = RobotFeedback(0, 0, ARM_JOINTS, np.zeros(7), "simulated_actuator_feedback",
                             "actuator_side", True)
    params = dict(timeout_s=0.01, model_semantics="actuator_side", residual_bound_nm=np.ones(7),
                  load_upper_n=5, force_limit_n=80, model_verified=True, stop_verified=True,
                  contact_consistent=True)
    return feedback, params


@pytest.mark.parametrize("change", [
    dict(torque_nm=None), dict(torque_nm=np.full(7, np.nan)), dict(semantics="unknown"),
    dict(source="commands"), dict(healthy=False), dict(available_s=0.1),
    dict(joint_names=ARM_JOINTS[::-1]), dict(units="N"),
])
def test_missing_invalid_or_unverified_torque_never_admits_load(change):
    feedback, params = load_inputs()
    assert admit_load(feedback, 0, ARM_JOINTS, **params).admitted
    assert not admit_load(replace(feedback, **change), 0, ARM_JOINTS, **params).admitted


def test_fresh_torque_alone_cannot_qualify_load_or_stop():
    feedback, params = load_inputs()
    assert not admit_load(feedback, 0.02, ARM_JOINTS, **params).admitted
    for change in (dict(timeout_s=None), dict(model_verified=False), dict(stop_verified=False),
                   dict(contact_consistent=False), dict(load_upper_n=None), dict(load_upper_n=81),
                   dict(residual_bound_nm=None)):
        assert not admit_load(feedback, 0, ARM_JOINTS, **(params | change)).admitted
    missing = RobotFeedback.unavailable(0, ARM_JOINTS)
    assert missing.torque_nm is None and not admit_load(missing, 0, ARM_JOINTS, **params).admitted
