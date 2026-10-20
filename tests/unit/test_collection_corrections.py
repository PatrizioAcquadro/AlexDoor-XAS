"""Executed error pulses and safety-gated response accounting."""

import numpy as np
import pytest

from alexdoor_xas.qualification.collection_corrections import INJECTED, CollectionCorrection
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
        p, rotation = case.target(np.zeros(3), np.eye(3), phase, np.zeros(3))
        assert np.isfinite(p).all() and np.isfinite(rotation).all()
        assert np.linalg.norm(p) <= 0.021
        case.observe(dict(valid=False, loaded=True, separation_m=0))
    assert case.injected_commands > 0 and case.corrective_commands > 0 and case.pending
    assert maximum > 1 if family == "C" and variant == 2 else maximum == 1
    for _ in range(31):
        case.role = 2
        case.observe(dict(valid=True, loaded=False, separation_m=0.02))
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
