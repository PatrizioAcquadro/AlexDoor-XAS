"""Executed error pulses and safety-gated response accounting."""

import numpy as np
import pytest

from alexdoor_xas.qualification.collection_corrections import (
    CORRECTIVE,
    EXPERT,
    INJECTED,
    CollectionCorrection,
)
from alexdoor_xas.qualification.synthetic_probe import ProbeSetup


@pytest.mark.parametrize("family", ["A", "B", "C", "D"])
@pytest.mark.parametrize("variant", [1, 2])
def test_every_correction_executes_a_bounded_pulse_and_requires_response(family, variant):
    setup = ProbeSetup()
    case = CollectionCorrection(family, f"correction-{variant}", setup, 1 / 60)
    phase = {
        "A": "approach",
        "B": "contact" if variant == 1 else "push",
        "C": "push",
        "D": "release",
    }[family]
    maximum = 0.0
    for _ in range(480):
        maximum = max(maximum, case.rate_scale())
        advance = case.injected_reference_advance(0.0001)
        injected_pose = None if advance is None else (np.array([0.3 * advance, 0, 0]), np.eye(3))
        if (gap := case.material_gap_target(phase)) is not None:
            injected_pose = (np.array([-gap, 0, 0]), np.eye(3))
        if case.holds_release_contact():
            injected_pose = (np.zeros(3), np.eye(3))
        p, rotation = case.target(
            np.zeros(3), np.eye(3), phase, np.zeros(3), injected_pose=injected_pose
        )
        assert np.isfinite(p).all() and np.isfinite(rotation).all()
        assert np.linalg.norm(p) <= 0.021
        case.observe(dict(valid=False, loaded=True, separation_m=0))
    assert case.injected_commands > 0 and case.corrective_commands > 0 and case.pending
    assert maximum > 1 if family == "C" and variant == 2 else maximum == 1
    for _ in range(31):
        case.role = 2
        case.observe(dict(valid=True, loaded=family != "D", separation_m=0.02))
    assert not case.pending
    unsafe = dict(
        physical_safe=False,
        footprint_inside=True,
        material_error=0,
        material_orientation_error=0,
        force=0,
    )
    assert not case.permit_recontact(unsafe, 0.5)
    assert case.role != INJECTED


def test_full_disturbance_survives_until_measured_corrective_handoff():
    from alexdoor_xas.qualification.collection_corrections import CORRECTIVE

    dt = 1 / 60
    case = CollectionCorrection("A", "correction-1", ProbeSetup(), dt)
    measured_p, measured_r = np.zeros(3), np.eye(3)
    delivered = []
    for _ in range(360):
        p, r = case.target(np.zeros(3), np.eye(3), "approach", measured_p, measured_r)
        # This orchestration test uses exact tracking to isolate the target/mask
        # mechanism; physical tracking and safety are separately qualified.
        measured_p, measured_r = p, r
        delivered.append(p.copy())
    np.testing.assert_allclose(delivered[-15:], np.tile([0, 0.02, 0], (15, 1)))
    assert case.role == INJECTED
    p, r = case.target(np.zeros(3), np.eye(3), "approach", measured_p, measured_r)
    assert case.role == CORRECTIVE
    assert case.corrective_start_error["position_m"] == pytest.approx(0.02)
    assert case.corrective_start_error["rotation_rad"] == pytest.approx(np.deg2rad(5))
    assert 0 < np.linalg.norm(p - measured_p) < 0.001
    row = dict(valid=False, loaded=False, separation_m=0)
    case.observe(row)
    reference = row["correction_reference"]
    assert reference["error"]["position_m"] == pytest.approx(0.02)
    np.testing.assert_allclose(reference["position_m"], 0)
    # The recorded error belongs to the pre-action state and ordinary goal,
    # rather than the tiny displacement of the first blended command.
    assert reference["error"]["position_m"] > 10 * np.linalg.norm(p - measured_p)
    for _ in range(30):
        measured_p, measured_r = p, r
        p, r = case.target(np.zeros(3), np.eye(3), "approach", measured_p, measured_r)
    np.testing.assert_allclose(p, 0, atol=1e-12)
    assert case.corrective_latest_error["position_m"] < 1e-12


def test_pace_pulse_changes_executed_reference_when_ordinary_clock_is_capped():
    from scipy.spatial.transform import Rotation

    from alexdoor_xas.qualification.collection_corrections import CORRECTIVE
    from alexdoor_xas.qualification.synthetic_probe import push_reference

    dt = 1 / 60
    setup = ProbeSetup()
    case = CollectionCorrection("C", "correction-2", setup, dt)
    reference = 0.0
    measured_p, measured_r = np.array([0.3, 0, 0]), np.eye(3)
    previous_requested = None
    injected_ratios = []
    last_advance = 0.0
    handed_off = False
    for tick in range(400):
        angle = 0.006 * tick * dt
        nominal = push_reference(reference, angle, setup, dt, np.pi / 2)
        if case.pulse("push") > 0:
            assert push_reference(reference, angle, setup, 1.5 * dt, np.pi / 2) == nominal
        step = nominal - reference
        reference = nominal
        scale = case.rate_scale()
        advance = case.injected_reference_advance(step)

        def pose(value):
            return np.array([0.3 * np.cos(value), 0.3 * np.sin(value), 0]), Rotation.from_euler(
                "z", value
            ).as_matrix()

        p, r = pose(reference)
        injected = None if advance is None else pose(reference + advance)
        prior_role = case.role
        p, r = case.target(p, r, "push", measured_p, measured_r, injected_pose=injected)
        requested = Rotation.from_matrix(r).as_euler("xyz")[2]
        if case.role == INJECTED:
            if prior_role == INJECTED and scale == 1.5:
                injected_ratios.append((requested - previous_requested) / step)
            last_advance = advance
        elif case.role == CORRECTIVE and prior_role == INJECTED:
            handed_off = True
            assert case.corrective_start_error["rotation_rad"] > 0.9 * last_advance
        previous_requested = requested
        measured_p, measured_r = p, r
    assert handed_off and last_advance > 0.002
    np.testing.assert_allclose(injected_ratios, 1.5, atol=1e-10, rtol=0)


def test_pace_injection_cannot_silently_omit_the_advanced_pose():
    case = CollectionCorrection("C", "correction-2", ProbeSetup(), 1 / 60)
    case.counts["push"] = 280
    with pytest.raises(ValueError, match="advanced reference pose"):
        case.target(np.zeros(3), np.eye(3), "push", np.zeros(3))


@pytest.mark.parametrize(
    "changed",
    [
        None,
        "physical_safe",
        "footprint_inside",
        "material_error",
        "material_orientation_error",
        "force",
        "duration",
        "injected",
    ],
)
def test_only_safe_bounded_corrective_recontact_can_be_supervised(changed):
    from alexdoor_xas.qualification.collection_corrections import CORRECTIVE

    case = CollectionCorrection("B", "correction-2", ProbeSetup(), 1 / 60)
    case.had_contact = case.pending = True
    case.unloaded_s = 0.4 if changed != "duration" else 1.1
    case.role = INJECTED if changed == "injected" else CORRECTIVE
    row = dict(
        valid=False,
        loaded=False,
        physical_safe=True,
        footprint_inside=True,
        material_error=0.001,
        material_orientation_error=0.01,
        force=0.0,
        separation_m=0.001,
        phase_valid=False,
    )
    if changed in ("physical_safe", "footprint_inside"):
        row[changed] = False
    elif changed == "material_error":
        row[changed] = 0.02
    elif changed == "material_orientation_error":
        row[changed] = 0.3
    elif changed == "force":
        row[changed] = 80.0
    case.observe(row)
    assert row["recovery_valid"] is (changed is None)
    assert row["phase_valid"] is (changed is None)
    assert case.pending  # closing motion never stands in for restored loaded support


def test_b2_material_gap_is_independent_of_the_ordinary_forward_lead():
    case = CollectionCorrection("B", "correction-2", ProbeSetup(), 1 / 60)
    case.counts["push"] = 75
    goal = np.array([0.0008, 0, 0])  # the ordinary goal is ahead of the current panel
    gap = case.material_gap_target("push")
    assert gap == 0.0015
    with pytest.raises(ValueError, match="material-gap pose"):
        case.target(goal, np.eye(3), "push", goal)
    p, _ = case.target(
        goal, np.eye(3), "push", goal, injected_pose=(np.array([-gap, 0, 0]), np.eye(3))
    )
    assert p[0] == pytest.approx(-0.0015)
    assert p[0] != pytest.approx(goal[0] - 0.0015)


def test_gap_catchup_waits_for_measured_position_return_to_finish():
    case = CollectionCorrection("C", "correction-1", ProbeSetup(), 1 / 60)
    case.counts["push"] = 280
    assert case.position_return_active("push")
    # Last injected command and its actual displaced state.
    p, r = case.target(np.zeros(3), np.eye(3), "push", np.zeros(3))
    case.counts["push"] = 300
    assert case.position_return_active("push")  # before first corrective dispatch
    for _ in range(7):
        p, r = case.target(np.zeros(3), np.eye(3), "hold", p, r)
        assert case.position_return_active("hold")
    for _ in range(9):
        p, r = case.target(np.zeros(3), np.eye(3), "hold", p, r)
    assert case.corrective_transition_complete
    assert not case.position_return_active("hold")
    assert case.pending  # transition completion alone is not physical recovery


def test_c1_measured_return_fits_recontact_budget_without_exceeding_gap_catchup_rate():
    case = CollectionCorrection("C", "correction-1", ProbeSetup(), 1 / 60)
    case.counts["push"] = 299
    case.target(np.zeros(3), np.eye(3), "push", np.array([-0.0009333347, 0, 0]))
    positions = [-0.0009333347]
    for _ in range(16):
        p, _ = case.target(np.zeros(3), np.eye(3), "push", np.array([positions[-1], 0, 0]))
        positions.append(p[0])
    assert max(np.diff(positions)) / case.dt <= 0.006
    assert positions[-2] == pytest.approx(0, abs=1e-10)
    assert case.corrective_transition_complete
    assert case.pending  # still needs measured sustained valid contact
    assert (
        CollectionCorrection("B", "correction-2", ProbeSetup(), 1 / 60).corrective_transition_s
        == 0.5
    )


@pytest.mark.parametrize("variant,last_injected", [("correction-1", 299), ("correction-2", 311)])
def test_late_handoff_requires_sustained_support_before_controller_transition(
    variant, last_injected
):
    case = CollectionCorrection("C", variant, ProbeSetup(), 1 / 60)
    case.counts["push"] = last_injected
    case.target(
        np.zeros(3),
        np.eye(3),
        "push",
        np.zeros(3),
        injected_pose=(np.array([0.001, 0, 0]), np.eye(3)),
    )
    case.target(np.zeros(3), np.eye(3), "push", np.array([-0.001, 0, 0]))
    row = dict(phase="push", valid=False, physical_safe=True)
    assert not case.needs_supported_hold_handoff(row)
    row.update(valid=True, loaded=True, separation_m=0)
    for _ in range(29):
        case.observe(row.copy())
        assert not case.needs_supported_hold_handoff(row)
    # A loss resets the same unchanged sustain predicate; first impact cannot
    # cancel the corrective return or change the position-to-force controller.
    case.observe(dict(row, valid=False))
    for _ in range(29):
        case.observe(row.copy())
        assert not case.needs_supported_hold_handoff(row)
    case.observe(row.copy())
    assert not case.pending
    assert case.needs_supported_hold_handoff(row)
    case.accept_supported_hold_handoff()
    assert not case.pending
    assert case.corrective_origin is None
    assert not case.needs_supported_hold_handoff(row)
    achieved = np.array([0.0001, 0.0002, 0.0003])
    goal, _ = case.target(achieved, np.eye(3), "hold", achieved)
    np.testing.assert_array_equal(goal, achieved)
    assert case.role == EXPERT  # sustained physical recovery has already completed


@pytest.mark.parametrize("variation", ["correction-1", "correction-2"])
def test_release_disturbance_tracks_moving_material_until_corrective_handoff(variation):
    case = CollectionCorrection("D", variation, ProbeSetup(), 1 / 60)
    case.counts["release"] = 60 if variation == "correction-1" else 30
    gap = case.material_gap_target("release")
    assert case.holds_release_contact() == (variation == "correction-2")
    rotation = np.eye(3)
    for material_origin in (np.zeros(3), np.array([0.004, 0, 0])):
        position = material_origin - np.array([gap or 0, 0, 0])
        delivered, _ = case.target(
            material_origin - [0.03, 0, 0],
            rotation,
            "release",
            position,
            injected_pose=(position, rotation),
        )
        np.testing.assert_allclose(delivered - material_origin, [-float(gap or 0), 0, 0])
        assert case.role == INJECTED  # no automatic unloading from material motion
    case.counts["release"] = 90 if variation == "correction-1" else 60
    case.target(material_origin - [0.03, 0, 0], rotation, "release", position)
    assert case.role == CORRECTIVE and case.pending
    assert case.corrective_start_error["position_m"] >= 0.022 - 1e-10


def test_c2_soft_return_keeps_full_residual_and_original_recontact_envelope():
    case = CollectionCorrection(
        "C", "correction-2", ProbeSetup(), 1 / 60, corrective_transition_s=0.75
    )
    case.counts["push"] = 311
    case.target(
        np.zeros(3),
        np.eye(3),
        "push",
        np.zeros(3),
        injected_pose=(np.array([0.00148, 0, 0]), np.eye(3)),
    )
    points = [np.array([0.00148, 0, 0])]
    for _ in range(45):
        target, _ = case.target(np.zeros(3), np.eye(3), "push", points[0])
        points.append(target)
    assert case.corrective_start_error["position_m"] == pytest.approx(0.00148)
    assert points[1][0] > 0.00147  # no excluded taper or automatic cancellation
    np.testing.assert_allclose(points[-1], 0, atol=1e-12)
    assert np.max(np.linalg.norm(np.diff(points, axis=0), axis=1)) * 60 < 0.0032
    assert case.pending  # actual contact, hold and stopping still decide success
    assert (
        CollectionCorrection("C", "correction-2", ProbeSetup(), 1 / 60).corrective_transition_s
        == 0.5
    )
    for invalid in (-1, 0, 1.01, float("nan")):
        with pytest.raises(ValueError):
            CollectionCorrection(
                "C", "correction-2", ProbeSetup(), 1 / 60, corrective_transition_s=invalid
            )


def test_pace_exit_needs_executed_pulse_and_real_contiguous_loaded_support():
    case = CollectionCorrection("C", "correction-2", ProbeSetup(), 1 / 60)
    rows = [dict(time=i / 60, valid=True, physical_safe=True) for i in range(31)]
    case.counts["push"] = 311
    case.role = INJECTED
    case.injected_commands = 59
    assert not case.supported_pace_exit(rows)  # pulse still active
    case.counts["push"] = 312
    assert case.supported_pace_exit(rows)
    assert not case.supported_pace_exit(rows[1:])  # no invented earlier support
    assert not case.supported_pace_exit([*rows[:-1], dict(rows[-1], valid=False)])
    assert not case.supported_pace_exit([*rows[:-1], dict(rows[-1], physical_safe=False)])
    case.accept_supported_hold_handoff()
    assert not case.supported_pace_exit(rows)  # only one carried-state transition


@pytest.mark.parametrize("kind", ["pause", "pose"])
def test_approach_neighborhood_preserves_error_history_and_measured_return(kind):
    d = dict(
        kind=kind,
        start_s=1.0,
        end_s=2.0,
        translation_m=[0.004, -0.006, 0],
        rotation_rad=[0, 0, 0.03],
    )
    case = CollectionCorrection("A", "correction-1", ProbeSetup(), 1 / 60, approach_disturbance=d)
    measured = np.array([0.1, 0.2, 0.3])
    for tick in range(120):
        p, r = case.target(
            np.array([tick / 1000, 0.2, 0.3]), np.eye(3), "approach", measured, np.eye(3)
        )
        measured = p
    assert case.role == INJECTED
    if kind == "pause":
        np.testing.assert_allclose(measured, case.pause_pose[0])
    before = measured.copy()
    p, r = case.target(np.array([0.12, 0.2, 0.3]), np.eye(3), "approach", measured)
    assert case.role == CORRECTIVE
    assert np.linalg.norm(p - before) < 0.001
    for _ in range(31):
        p, r = case.target(np.array([0.12, 0.2, 0.3]), np.eye(3), "approach", p, r)
    np.testing.assert_allclose(p, [0.12, 0.2, 0.3])


def test_approach_disturbance_cannot_consume_corrective_return_reserve():
    with pytest.raises(ValueError, match="complete measured return"):
        CollectionCorrection(
            "A",
            "correction-1",
            ProbeSetup(),
            1 / 60,
            approach_disturbance=dict(kind="pause", start_s=5, end_s=6),
        )
