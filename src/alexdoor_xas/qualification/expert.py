"""Paired expert decision and publication, independent of simulator startup."""

import json
from pathlib import Path

import numpy as np

from .synthetic_probe import summarize_trials


def qualify_pair(trials):
    """Only a complete, controlled pair establishes an expert reference."""
    if len(trials) != 2:
        return dict(status="unresolved", theta_expert_d=None, reason="requires_two_trials")
    if any(
        not r.get("passed")
        or not r.get("released")
        or r.get("hold_angle_deg") is None
        or not np.isfinite(r["hold_angle_deg"])
        for r in trials
    ):
        return dict(status="unresolved", theta_expert_d=None, reason="invalid_controlled_cycle")
    result = summarize_trials(trials, required_count=2)
    if not result["passed"]:
        return dict(
            status="unresolved",
            theta_expert_d=None,
            reason="inconsistent_or_unresolved_pair",
            repeat_spread_deg=result["repeat_spread_deg"],
        )
    return dict(
        status="qualified" if result["meets_45_deg"] else "out_of_domain",
        theta_expert_d=result["angle_deg"],
        repeat_spread_deg=result["repeat_spread_deg"],
        reason=trials[0]["stop_reason"],
    )


def write_result(path, value):
    """Publish a complete JSON record without a partially written destination."""
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def continue_pair(output, folder, config, device):
    """Restore only an untouched first trial across the runner's process restart."""
    output, folder = Path(output), Path(folder)
    for name in ("prepared.json", "recipe.json", "candidate.json"):
        if (output / name).read_bytes() != (folder / name).read_bytes():
            raise ValueError(f"Pair input changed before second trial: {name}")
    if (output / "setup.json").read_bytes() != Path(config).read_bytes():
        raise ValueError("Pair setup changed before second trial")
    report = json.loads((output / "report.json").read_text())
    trial = json.loads((output / "repeat-1/result.json").read_text())
    if (
        report["asset_id"] != folder.name
        or report["device"] != device
        or report["diagnostic_only"]
        or report["status"] != "unresolved"
        or report["reason"] != "incomplete_execution"
        or report["trials"] != [trial_summary(trial, 1)]
        or (output / "repeat-2").exists()
    ):
        raise ValueError("Continuation requires exactly one completed trial and no second attempt")
    return report, [trial]


def publish_expert(folder, original, report, output):
    path = Path(folder) / "prepared.json"
    current = json.loads(path.read_text())
    if current != original:
        raise RuntimeError("Prepared record changed during qualification; result remains in cache")
    current["expert_qualification"] = {
        key: report[key] for key in ("status", "theta_expert_d", "reason")
    }
    current["expert_qualification"].update(
        evidence=str(output),
        trials=report["trials"],
        repeat_spread_deg=report.get("repeat_spread_deg"),
        scope="nominal_simulated_expert_reference_not_learning_data",
    )
    write_result(path, current)


def probe_config(door, setup, device, cameras=True):
    from alexdoor_xas.assets.synthetic_door import SyntheticDoor
    from alexdoor_xas.envs.door_task.door_push_purdue_env_cfg import DoorPushPurdueEnvCfg

    cfg = DoorPushPurdueEnvCfg()
    cfg.sim.device = device
    cfg.cameras = cameras
    if isinstance(door, SyntheticDoor):
        cfg.synthetic_door = door
    else:
        cfg.prepared_door = door
    cfg.floor_pose = tuple(setup.floor_pose)
    cfg.initial_joints = {
        **cfg.initial_joints,
        **setup.initial_joints,
        "NECK_Z": setup.neck[0],
        "NECK_Y": setup.neck[1],
    }
    cfg.max_joint_speed = setup.max_joint_speed
    cfg.centering_gain = setup.centering_gain
    cfg.ik_damping = setup.ik_damping
    return cfg


def trial_summary(trial, number):
    """Compact per-door result; detailed traces live beside the report."""
    keys = (
        "passed",
        "angle_deg",
        "hold_angle_deg",
        "stop_reason",
        "safety_detail",
        "peak_force_n",
        "joint_margin",
        "released",
        "visibility",
    )
    result = {key: trial.get(key) for key in keys}
    result["directory"] = f"repeat-{number}"
    return result


def valid_reference_inputs(record):
    return (
        record.get("status") == "ready_for_5.1"
        and record.get("initial_state") == "closed_unlatched"
        and all(
            record.get("checks", {}).get(k) == "pass"
            for k in ("normalization", "static", "visual", "physics")
        )
        and np.isfinite(record.get("mechanical_limit_deg", np.nan))
    )
