"""Maintained axis-free contact contracts, independent of a tracker implementation."""

from dataclasses import replace

import numpy as np
import pytest

from alexdoor_xas.action.frames import ObjectFrame
from alexdoor_xas.perception.contracts import (
    DoorEstimate,
    FieldSupport,
    LocalContactSelection,
    LocalMaterialState,
    LocalPatchState,
    validate_local_contact,
    validate_local_transition,
)


def supported_patch(patch_id, time=1.0):
    support = FieldSupport(0, time, time, 0, 0.004, 0.02)
    return LocalPatchState(
        patch_id,
        ObjectFrame(np.array([1, 0, 1.1]), np.eye(3)),
        support,
        support,
        support,
        (0.04, 0.04),
        "tracked_material",
    )


def test_visual_reference_and_explicit_contact_have_independent_lifetimes():
    reference, contact = supported_patch("visual"), supported_patch("contact", 0.5)
    selection = LocalContactSelection("s1", "contact", 0.5, "diagnostic")
    state = LocalMaterialState(0, "visual", (reference, contact), selection)
    with pytest.raises(ValueError, match="stale_dynamic_support"):
        validate_local_contact(state, 1)
    # A stale visual reference does not erase a separately supported selected patch.
    state = replace(state, patches=(supported_patch("visual", 0.5), supported_patch("contact")))
    assert validate_local_contact(state, 1).patch_id == "contact"
    assert not DoorEstimate(1, False, "local", local=state).valid


def test_selection_requires_source_identity_time_and_explicit_predecessor():
    previous = LocalContactSelection("s1", "high", 1, "diagnostic")
    validate_local_transition(None, previous)
    with pytest.raises(ValueError, match="reselected"):
        validate_local_transition(previous, replace(previous, patch_id="low"))
    with pytest.raises(ValueError, match="missing_contact_transition"):
        validate_local_transition(previous, LocalContactSelection("s2", "low", 2, "diagnostic"))
    following = LocalContactSelection("s2", "low", 2, "diagnostic", "s1")
    validate_local_transition(previous, following)
    state = LocalMaterialState(0, "high", (supported_patch("low", 2),), following)
    assert validate_local_contact(state, 2).patch_id == "low"
    with pytest.raises(ValueError, match="missing_explicit"):
        validate_local_contact(replace(state, selection=replace(following, source="tracker")), 2)


def test_local_generation_rejects_boolean_episode_identity():
    state = LocalMaterialState(
        False,
        "contact",
        (supported_patch("contact"),),
        LocalContactSelection("s", "contact", 1, "diagnostic"),
    )
    with pytest.raises(ValueError, match="wrong_episode_generation"):
        validate_local_contact(state, 1, generation=0)


def test_field_gates_allow_roundoff_without_an_extra_tick_of_dynamic_freshness():
    from alexdoor_xas.perception.contracts import POSITION_LIMIT_M, ROTATION_LIMIT_RAD

    support = FieldSupport(0, 0, 0, 0, POSITION_LIMIT_M + 1e-7, ROTATION_LIMIT_RAD + 1e-7)
    support.require(0.15 + 1e-12, 0, dynamic=True)
    support.require_bounds(position=True, rotation=True, qualification=True)
    with pytest.raises(ValueError, match="stale_dynamic_support"):
        support.require(0.15 + 1 / 60, 0, dynamic=True)
    with pytest.raises(ValueError, match="excessive"):
        replace(support, position_bound_m=0.011).require_bounds(position=True, qualification=True)
