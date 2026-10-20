"""Teacher phase flags cannot substitute for actual subtask contact and motion."""

from dataclasses import replace

import numpy as np
import pytest

from alexdoor_xas.qualification.collection_phase import (
    evaluate_phase_endpoint,
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
