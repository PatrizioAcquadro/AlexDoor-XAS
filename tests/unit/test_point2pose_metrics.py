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


def test_physical_targets_use_actual_seed_and_acquisition_camera():
    import numpy as np

    from alexdoor_xas.action.frames import rot_z
    from alexdoor_xas.perception.diagnostics.targets import evaluate_targets

    cameras = np.tile(np.eye(4), (3, 1, 1))
    cameras[:, 0, 3] = [0, 0.3, -0.2]
    angles = np.array([0, 0.2, 0.5])
    h5 = {
        "observations/camera_world": cameras,
        "annotations/hinge_origin": np.zeros((3, 3)),
        "annotations/hinge_rotation": np.tile(np.eye(3), (3, 1, 1)),
        "annotations/signed_angle": angles,
    }
    targets = [np.eye(4) for _ in range(3)]
    for i, target in enumerate(targets):
        target[:3, 3] = [1, i * 0.1, 0.5]
    frames = []
    for row in (1, 2):
        motion = np.eye(4)
        motion[:3, :3] = rot_z(angles[row] - angles[1])
        pose = np.linalg.inv(cameras[row]) @ motion @ cameras[1]
        frames.append(
            dict(
                row=row,
                time_s=float(row),
                is_seed=row == 1,
                objects=[
                    dict(
                        object_id=0,
                        camera_from_map=pose,
                        native_tracking=True,
                        native_lost=False,
                        integration_accepted=True,
                    )
                ],
            )
        )
    scores = evaluate_targets(h5, frames, targets, reference_row=0, seed_row=1)
    assert len(scores) == 3
    for result in scores:
        assert result["metrics"]["accepted_errors"]["position_m"]["maximum"] < 1e-10
        assert result["metrics"]["accepted_errors"]["rotation_deg"]["maximum"] < 1e-6
        assert result["metrics"]["thresholds"]["10"]["availability"] == 0.5
    missing = evaluate_targets(h5, frames, targets, reference_row=0, seed_row=None)
    assert all(r["metrics"]["thresholds"]["20"]["availability"] == 0 for r in missing)
