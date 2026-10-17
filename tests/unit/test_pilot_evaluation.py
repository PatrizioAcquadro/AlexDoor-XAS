"""Independent evaluator gates cannot turn coasting or a stop into task completion."""

import numpy as np
import pytest

from alexdoor_xas.qualification.pilot_evaluation import PilotEvaluation
from alexdoor_xas.qualification.synthetic_probe import ProbeSetup


def row(t, **changed):
    return (
        dict(
            time_s=t,
            opening_angle_rad=np.deg2rad(45),
            native_force_n=[1.0, 1.0],
            native_gap_m=[0.0, 0.0],
            physical_violations=[],
            footprint_inside=True,
            material_error_m=0.0,
            material_orientation_error_rad=0.0,
            learned_command=True,
            surface_separation_m=0.0,
            applied_withdrawal_m=-0.001,
            applied_tracking_error_m=0.0,
            commanded_withdrawal=False,
        )
        | changed
    )


def holding():
    score = PilotEvaluation(ProbeSetup(), 50.0, 1 / 60)
    for tick in range(1, 211):
        score.update(row(tick / 60))
    assert score.result()["hold_valid"]
    return score


def release(score, *, learned=True):
    for tick in range(211, 231):
        score.update(
            row(
                tick / 60,
                native_force_n=[0.0, 0.0],
                native_gap_m=[None, None],
                learned_command=learned,
                commanded_withdrawal=True,
                applied_withdrawal_m=0.02,
                surface_separation_m=0.02,
            )
        )


def test_complete_controlled_hold_and_commanded_release_pass():
    score = holding()
    release(score)
    result = score.result()
    assert result["task_success"] and result["released"]
    assert result["controlled_fraction_of_expert"] == pytest.approx(0.9)
    assert result["contact_losses"] == 0


def test_free_coasting_cannot_supply_controlled_opening_or_hold():
    score = PilotEvaluation(ProbeSetup(), 50.0, 1 / 60)
    for tick in range(1, 211):
        score.update(
            row(
                tick / 60,
                opening_angle_rad=np.deg2rad(70),
                native_force_n=[0.0, 0.0],
                native_gap_m=[None, None],
            )
        )
    release(score)
    assert not score.result()["task_success"]
    assert score.result()["controlled_opening_deg"] == 0
    assert not score.result()["hold_valid"]


def test_frozen_stop_cannot_become_a_commanded_release():
    score = holding()
    release(score, learned=False)
    assert not score.result()["released"] and not score.result()["task_success"]


def test_one_native_gap_or_safety_violation_preserves_failure():
    score = holding()
    score.update(
        row(211 / 60, native_gap_m=[0.0, 0.0002], physical_violations=["forbidden_contact"])
    )
    assert score.last_row["loaded"] is False
    assert score.result()["contact_losses"] == 1
    assert score.result()["physical_violations"] == ["forbidden_contact"]
    with pytest.raises(ValueError, match="clock gap"):
        score.update(row(213 / 60))


def test_expected_unloading_may_follow_an_earlier_withdrawal_command():
    score = holding()
    score.update(row(211 / 60, commanded_withdrawal=True))
    for tick in range(212, 232):
        score.update(
            row(
                tick / 60,
                native_force_n=[0.0, 0.0],
                native_gap_m=[None, None],
                surface_separation_m=0.02,
                commanded_withdrawal=False,
            )
        )
    assert score.result()["released"] and score.result()["contact_losses"] == 0


def test_withdrawal_before_complete_hold_cannot_hide_later_contact_loss():
    score = PilotEvaluation(ProbeSetup(), 50.0, 1 / 60)
    for tick in range(1, 121):
        score.update(row(tick / 60, commanded_withdrawal=tick == 119))
    assert not score.release_attempted
    score.update(row(121 / 60, native_force_n=[0.0, 0.0], native_gap_m=[None, None]))
    assert score.result()["contact_losses"] == 1


def test_missing_native_feedback_is_explicitly_unavailable():
    score = holding()
    with pytest.raises(ValueError, match="two native"):
        score.update(row(211 / 60, native_force_n=[], native_gap_m=[]))
