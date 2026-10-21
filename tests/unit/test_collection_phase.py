"""Teacher phase flags cannot substitute for actual subtask contact and motion."""

from dataclasses import replace

import numpy as np
import pytest

from alexdoor_xas.qualification.collection_phase import (
    correction_hold_complete,
    evaluate_phase_endpoint,
    qualify_opening_requirement,
    qualify_phase_result,
)
from alexdoor_xas.qualification.synthetic_probe import ProbeSetup


def rows(phase, *, start=0, count=61, loaded=True, motion=0.0, speed=0.0):
    return [
        dict(
            time=start + i / 60,
            phase=phase,
            physical_safe=True,
            loaded=loaded,
            footprint_inside=True,
            material_error=0.001,
            material_orientation_error=0.001,
            angle=motion * i / 60,
            speed=speed,
            separation_m=0.02,
            tool_position=[0, 0, 0],
        )
        for i in range(count)
    ]


def test_acquisition_needs_actual_sustained_support():
    setup = ProbeSetup()
    assert evaluate_phase_endpoint(rows("contact"), "A", setup)["passed"]
    assert not evaluate_phase_endpoint(rows("contact", loaded=False), "A", setup)["passed"]
    assert not evaluate_phase_endpoint(rows("contact", count=30), "A", setup)["passed"]
    perturbed = rows("contact")
    for row in perturbed:
        row["material_error"] = setup.position_tolerance + 1e-4
    assert not evaluate_phase_endpoint(perturbed, "A", setup)["passed"]


def test_opening_requires_controlled_motion_beyond_acquisition():
    setup = ProbeSetup()
    contact = rows("contact")
    push = rows("push", start=61 / 60, motion=np.deg2rad(4))
    assert evaluate_phase_endpoint(contact + push, "B", setup)["passed"]
    assert not evaluate_phase_endpoint(contact + rows("push", start=61 / 60), "B", setup)["passed"]
    for row in push:
        row["loaded"] = False
    assert not evaluate_phase_endpoint(contact + push, "B", setup)["passed"]


def test_push_to_hold_requires_motion_then_supported_settling():
    setup = replace(ProbeSetup(), release_s=6)
    push = rows("push", motion=np.deg2rad(2))
    hold = rows("hold", start=61 / 60)
    assert evaluate_phase_endpoint(push + hold, "C", setup)["passed"]
    for row in hold:
        row["speed"] = np.deg2rad(2)
    assert not evaluate_phase_endpoint(push + hold, "C", setup)["passed"]
    assert not evaluate_phase_endpoint(rows("push") + hold, "C", setup)["passed"]


def test_release_needs_prior_support_and_actual_unloading():
    setup = ProbeSetup()
    tail = rows("stabilize", start=61 / 60, loaded=False)
    assert evaluate_phase_endpoint(rows("hold") + tail, "D", setup)["passed"]
    assert not evaluate_phase_endpoint(tail, "D", setup)["passed"]
    for row in tail:
        row["separation_m"] = 0.009
    assert not evaluate_phase_endpoint(rows("hold") + tail, "D", setup)["passed"]


def test_b_endpoint_does_not_require_a_final_window_in_excluded_cleanup():
    trace = rows("contact") + rows("push", start=2, motion=np.deg2rad(4))
    cleanup = rows("hold", start=4, count=180)
    cleanup[-3]["loaded"] = cleanup[-2]["loaded"] = False
    trace += cleanup + rows("stabilize", start=7, loaded=False)
    original = dict(
        passed=False,
        stop_reason="invalid_hold",
        physical_violations=[],
        released=True,
        stabilized=True,
    )
    result = qualify_phase_result(trace, "B", ProbeSetup(), original)
    assert result["passed"] and result["subtask_qualified"]
    assert not result["full_task_success"]
    assert result["cleanup_hold_review"]["hold_rows_preserved"] == 180
    assert not result["cleanup_hold_review"]["final_hold_window_qualified"]
    assert original["passed"] is False and original["stop_reason"] == "invalid_hold"
    assert not qualify_phase_result(trace, "C", ProbeSetup(), original)["passed"]
    pending = dict(injected_commands=1, corrective_commands=1, pending_correction=True)
    assert not qualify_phase_result(trace, "B", ProbeSetup(), original, pending)["passed"]
    no_opening = rows("contact") + rows("push", start=2)
    assert not qualify_phase_result(no_opening, "B", ProbeSetup(), original)["passed"]
    trace[0]["physical_safe"] = False
    assert not qualify_phase_result(trace, "B", ProbeSetup(), original)["passed"]


@pytest.mark.parametrize(
    "failure", ["lost_hold_contact", "actual_joint_speed_limit", "unsafe_release"]
)
def test_b_cleanup_review_never_overrides_other_failures(failure):
    trace = rows("contact") + rows("push", start=2, motion=np.deg2rad(4))
    original = dict(
        passed=False,
        stop_reason=failure,
        physical_violations=[],
        released=True,
        stabilized=True,
    )
    assert not qualify_phase_result(trace, "B", ProbeSetup(), original)["passed"]


def test_pace_hold_waits_for_current_settling_after_full_minimum():
    setup = ProbeSetup()
    limit = np.deg2rad(setup.tie_deg) / setup.stabilization_window_s
    push = rows("push", count=301, motion=np.deg2rad(0.6))
    hold = rows("hold", start=301 / 60, count=180, speed=1.8 * limit)
    for row, speed in zip(hold[-30:], np.linspace(1.1 * limit, 0.9 * limit, 30), strict=True):
        row["speed"] = speed
    # The terminal value is already below the speed bound; its full window is not.
    assert not correction_hold_complete(push + hold, setup, 1 / 60)
    settled = rows("hold", start=481 / 60, count=31, speed=0.9 * limit)
    assert correction_hold_complete(push + hold + settled, setup, 1 / 60)
    later = rows("hold", start=512 / 60, count=2, speed=1.1 * limit)
    assert not correction_hold_complete(push + hold + settled + later, setup, 1 / 60)
    assert not correction_hold_complete(
        push + rows("hold", start=301 / 60, count=60), setup, 1 / 60
    )
    settled[-1]["loaded"] = False
    assert not correction_hold_complete(push + hold + settled, setup, 1 / 60)
    settled[-1]["loaded"] = True
    for i, row in enumerate(settled):
        row["tool_position"] = [0.003 * i / 30, 0, 0]
    assert not correction_hold_complete(push + hold + settled, setup, 1 / 60)


def test_opening_requirement_preserves_the_primary_physical_failure():
    failed = dict(passed=False, angle_deg=36, stop_reason="lost_contact")
    reviewed = qualify_opening_requirement(failed, 41)
    assert not reviewed["passed"] and not reviewed["opening_requirement_passed"]
    assert reviewed["stop_reason"] == "lost_contact"
    assert "required_opening_deg" not in failed
    insufficient = qualify_opening_requirement(
        dict(passed=True, angle_deg=36, stop_reason="held"), 41
    )
    assert not insufficient["passed"]
    assert insufficient["stop_reason"] == "required_opening_not_reached"
    enough = qualify_opening_requirement(dict(passed=True, angle_deg=42, stop_reason="held"), 41)
    assert enough["passed"] and enough["opening_requirement_passed"]


def test_release_preparation_requires_current_supported_task_goal_and_genuine_prefix():
    from alexdoor_xas.qualification.collection_phase import release_preparation_ready
    from alexdoor_xas.qualification.synthetic_probe import ProbeSetup

    setup = ProbeSetup()
    goal = np.deg2rad(41)
    rows = [
        dict(time=i / 60, phase="push", valid=True, physical_safe=True, angle=goal + 0.01)
        for i in range(84)
    ]
    options = dict(opening_goal_rad=goal, required_history_s=4.4, dt=1 / 60)
    assert release_preparation_ready(rows, setup, **options)
    assert not release_preparation_ready(rows[:-1], setup, **options)
    # An old supported window cannot authorize release after current loss,
    # physical failure, or failure to meet the original opening requirement.
    for change in ({"valid": False}, {"physical_safe": False}, {"angle": goal - 0.001}):
        changed = [*rows[:-1], dict(rows[-1], **change)]
        assert not release_preparation_ready(changed, setup, **options)
    assert setup.hold_settle_s == 3.0  # actual ordinary hold still runs in full


def test_opening_hold_needs_current_goal_window_and_original_minimum():
    from alexdoor_xas.qualification.collection_phase import opening_hold_complete
    from alexdoor_xas.qualification.synthetic_probe import SustainedAngle

    held = SustainedAngle(0.5)
    goal = np.deg2rad(43.52440513489278)
    for tick in range(181):
        held.update(tick / 60, goal + np.deg2rad((tick / 60 - 2.9) * 0.5), True)
    # The terminal angle exceeds the goal; the current valid window still does not.
    assert held.samples[-1][1] > goal
    assert not opening_hold_complete(held.current, 3, 3, goal)
    for tick in range(181, 205):
        held.update(tick / 60, goal + np.deg2rad((tick / 60 - 2.9) * 0.5), True)
    assert opening_hold_complete(held.current, 3.4, 3, goal)
    assert not opening_hold_complete(held.current, 2.9, 3, goal)
    held.update(205 / 60, goal + 0.01, False)
    assert not opening_hold_complete(held.current, 3.5, 3, goal)
