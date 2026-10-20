"""Teacher phase flags cannot substitute for actual subtask contact and motion."""

from dataclasses import replace

import numpy as np

from alexdoor_xas.qualification.collection_phase import evaluate_phase_endpoint
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
