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
