"""Action-space development metrics and conservative matched-pilot stopping."""

from copy import deepcopy

import numpy as np
from scipy.spatial.transform import Rotation

from alexdoor_xas.action.spaces import A1_JOINT_DELTA, A2_EE_DELTA


def action_errors(space, predicted, target):
    """Compare inferred physical commands, independently of training objectives."""
    predicted, target = np.asarray(predicted), np.asarray(target)
    width = 7 if space == A1_JOINT_DELTA else 6
    if (
        space not in (A1_JOINT_DELTA, A2_EE_DELTA)
        or predicted.shape != target.shape
        or predicted.ndim != 2
        or predicted.shape[1] != width
        or not np.isfinite(predicted).all()
        or not np.isfinite(target).all()
    ):
        raise ValueError("Expected finite matching A1/A2 command matrices")
    error = predicted - target
    if space == A1_JOINT_DELTA:
        joint = np.linalg.norm(error, axis=1)
        return joint, {"joint_l2_rad": joint}
    translation = np.linalg.norm(error[:, :3], axis=1)
    rotation = (
        Rotation.from_rotvec(predicted[:, 3:]) * Rotation.from_rotvec(target[:, 3:]).inv()
    ).magnitude()
    return np.maximum(translation / 0.01, rotation / np.deg2rad(5)), {
        "translation_m": translation,
        "rotation_rad": rotation,
    }


def update_selection(previous, candidate, config):
    """One cell's paired development decision; caller stops matched rounds on any stop."""
    if not np.isfinite(candidate["primary"]) or candidate["primary"] < 0:
        raise ValueError("Invalid development score")
    if previous is None:
        return dict(
            best=deepcopy(candidate),
            latest=deepcopy(candidate),
            no_progress=0,
            regressions=0,
            accepted=True,
            stop=False,
        )
    best = previous["best"]
    if set(candidate["bucket_p95"]) != set(best["bucket_p95"]):
        raise ValueError("Development suite changed")
    ratios = [
        candidate["bucket_p95"][key] / max(best["bucket_p95"][key], 1e-12)
        for key in best["bucket_p95"]
    ]
    if not np.isfinite(ratios).all() or any(x < 0 for x in ratios):
        raise ValueError("Invalid bucket errors")
    improvement = (
        candidate["primary"] <= best["primary"] * (1 - config["minimum_relative_improvement"])
        and candidate["primary"] < best["primary"]
    )
    guarded = max(ratios) <= 1 + config["maximum_bucket_p95_increase_for_selection"]
    accepted = bool(improvement and guarded)
    regression = (
        candidate["primary"] > best["primary"] * (1 + config["persistent_regression_relative"])
        or max(ratios) > 1 + config["persistent_bucket_p95_regression_relative"]
    )
    no_progress = 0 if accepted else previous["no_progress"] + 1
    regressions = previous["regressions"] + 1 if regression else 0
    return dict(
        best=deepcopy(candidate if accepted else best),
        latest=deepcopy(candidate),
        no_progress=no_progress,
        regressions=regressions,
        accepted=accepted,
        stop=no_progress >= config["no_progress_patience"]
        or regressions >= config["regression_patience"],
    )
