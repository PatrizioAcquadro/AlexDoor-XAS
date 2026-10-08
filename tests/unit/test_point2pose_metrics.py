"""Sensitivity bounds must not hide wrong acceptances or failed chronological tails."""

from alexdoor_xas.perception.diagnostics.metrics import evaluate_bounds


def test_seed_thresholds_rotation_and_missing_tail():
    rows = [
        dict(
            time_s=i / 60,
            is_seed=i == 0,
            native_pose_valid=True,
            integration_accepted=True,
            capture_error=[position, rotation],
        )
        for i, (position, rotation) in enumerate(
            [
                (0, 0),
                (0.009, 4),
                (0.012, 4),
                (0.018, 4),
                (0.019, 6),
            ]
        )
    ]
    rows += [dict(time_s=i / 60) for i in (5, 6)]
    result = evaluate_bounds(rows)
    assert result["scheduled"] == 7 and result["accepted_nonseed"] == 4
    for bound, correct in (("10", 1), ("15", 2), ("20", 3)):
        report = result["thresholds"][bound]
        assert report["correct"] == correct
        assert report["accepted_precision"] == correct / 4
        assert report["availability"] == correct / 7
        assert report["continuity"]["intervals"][0]["first_s"] == 0
        assert report["continuity"]["terminal"]["censored"]
    assert result["thresholds"]["20"]["continuity"]["terminal"]["first_s"] == 4 / 60
    assert result["accepted_errors"]["position_peak"]["time_s"] == 4 / 60
    assert sum(result["accepted_errors"]["position_histogram_mm"]["counts"]) == 4


def test_empty_and_no_acceptances():
    assert evaluate_bounds([])["thresholds"]["10"]["accepted_precision"] is None
    result = evaluate_bounds([dict(time_s=1), dict(time_s=2)])
    assert result["thresholds"]["20"]["continuity"]["terminal"]["samples"] == 2
