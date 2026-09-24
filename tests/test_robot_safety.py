"""Start-pose settle safety contracts."""

from __future__ import annotations

import pytest

from alexdoor_xas.kinematics.settle import StartPoseError, validate_start_pose_settle


def test_realized_pose_within_tolerance_passes_and_records() -> None:
    report = validate_start_pose_settle(
        (0.5, 0.2, 0.1),
        (0.503, 0.2, 0.1),
        settle_ticks_used=12,
        max_settle_ticks=90,
        tolerance_m=0.01,
    )
    assert report.to_dict() == {
        "requested_pos_m": [0.5, 0.2, 0.1],
        "realized_pos_m": [0.503, 0.2, 0.1],
        "residual_m": pytest.approx(0.003),
        "tolerance_m": 0.01,
        "settle_ticks_used": 12,
        "max_settle_ticks": 90,
        "passed": True,
        "orientation_checked": False,
    }


def test_excessive_residual_fails_closed() -> None:
    with pytest.raises(StartPoseError, match="residual 0.0300 m exceeds") as raised:
        validate_start_pose_settle(
            (0.5, 0.2, 0.1),
            (0.53, 0.2, 0.1),
            settle_ticks_used=90,
            max_settle_ticks=90,
            tolerance_m=0.01,
        )
    assert raised.value.report.to_dict() == {
        "requested_pos_m": [0.5, 0.2, 0.1],
        "realized_pos_m": [0.53, 0.2, 0.1],
        "residual_m": pytest.approx(0.03),
        "tolerance_m": 0.01,
        "settle_ticks_used": 90,
        "max_settle_ticks": 90,
        "passed": False,
        "orientation_checked": False,
    }


def test_non_finite_realized_pose_fails_closed() -> None:
    with pytest.raises(StartPoseError):
        validate_start_pose_settle(
            (0.5, 0.2, 0.1),
            (float("nan"), 0.2, 0.1),
            settle_ticks_used=1,
            max_settle_ticks=90,
            tolerance_m=0.01,
        )


def test_bad_tolerance_rejected() -> None:
    with pytest.raises(ValueError, match="tolerance_m"):
        validate_start_pose_settle(
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0),
            settle_ticks_used=0,
            max_settle_ticks=90,
            tolerance_m=0.0,
        )
