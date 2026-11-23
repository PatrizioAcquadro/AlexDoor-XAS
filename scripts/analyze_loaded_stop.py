"""Summarize all samples from fixed hold/stop trials; no fitting or exclusions."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from alexdoor_xas.qualification.stop import LOADED_STOP_LIMITS  # noqa: E402


def analyze(path):
    report = json.loads((path / "report.json").read_text())
    cases = [event["case"] for event in report["events"]]
    if not report["completed"] or sorted(cases) != ["loaded", "unloaded"]:
        raise ValueError(f"Incomplete loaded and unloaded stop trials in {path}")
    protocol = path / "protocol.json"
    criteria = (
        json.loads(protocol.read_text())["criteria"] if protocol.exists() else LOADED_STOP_LIMITS
    )
    for key in LOADED_STOP_LIMITS:
        if not np.isfinite(criteria[key]) or criteria[key] < 0:
            raise ValueError(f"Invalid stop criterion {key} in {path}")
    # Each phase uses the same channels. Decompress them once per run.
    with np.load(path / "trace.npz") as trace:
        data = dict(trace)
    calibration = report["runtime"]["calibration"]
    limits = np.array(report["runtime"]["effort_limits"])
    sign = calibration["sign"]
    applied_key = "applied" if "applied" in data else "applied_approx"
    phases = {}
    for phase in np.unique(data["phase"]):
        mask = data["phase"] == phase
        samples = np.flatnonzero(mask)
        steady = samples[-60:]
        estimates = data["rejection"][mask] == ""
        residual = sign * (data["torque"][mask] - data["expected"][mask])
        external = np.einsum("nji,nj->ni", data["jacobian"][mask], data["reference"][mask])
        bound = np.array(calibration["residual_bound_nm"])
        err = np.abs(residual - external)
        phases[str(phase)] = dict(
            samples=len(samples),
            available=int(sum(estimates)),
            rejections={
                str(k): int(sum(data["rejection"][mask] == k))
                for k in np.unique(data["rejection"][mask])
                if k
            },
            tracking_max_rad=np.max(
                np.abs(data["q"][mask] - data["target"][mask]), axis=0
            ).tolist(),
            steady_tracking_max_rad=float(
                np.max(np.abs(data["q"][steady] - data["target"][steady]))
            ),
            speed_max_rad_s=float(np.max(np.abs(data["dq"][mask]))),
            acceleration_max_rad_s2=float(np.max(np.abs(data["ddq"][mask]))),
            final_speed_rad_s=float(np.max(np.abs(data["dq"][samples[-1]]))),
            torque_max_nm=np.max(np.abs(data["computed"][mask]), axis=0).tolist(),
            applied_max_nm=np.max(np.abs(data[applied_key][mask]), axis=0).tolist(),
            clipping_samples=int(
                np.sum(
                    np.any(np.abs(data["computed"][mask] - data[applied_key][mask]) > 1e-5, axis=1)
                )
            ),
            max_effort_limit_fraction=float(np.max(np.abs(data[applied_key][mask]) / limits)),
            corrected_residual_max_nm=np.max(err, axis=0).tolist(),
            residual_exceedance_samples=int(sum(np.any(err > bound, axis=1))),
            force_error_max_n=(
                float(
                    np.max(
                        np.linalg.norm(
                            data["wrench"][mask][estimates, :3]
                            - data["reference"][mask][estimates, :3],
                            axis=1,
                        )
                    )
                )
                if any(estimates)
                else None
            ),
            upper_covers_reference=(
                bool(
                    np.all(
                        data["upper"][mask][estimates]
                        >= np.linalg.norm(data["reference"][mask][estimates, :3], axis=1)
                    )
                )
                if any(estimates)
                else None
            ),
        )

    def motion(phase, anchor):
        m = data["phase"] == phase
        distance = np.linalg.norm(data["position"][m] - anchor[0], axis=1) * 1000
        rotation = (
            Rotation.from_quat(data["quaternion"][m]) * Rotation.from_quat(anchor[1]).inv()
        ).magnitude()
        speeds = np.max(np.abs(data["dq"][m]), axis=1)
        persistent = np.maximum.accumulate(speeds[::-1])[::-1]
        settled = np.flatnonzero(persistent <= criteria["final_speed_rad_s"])
        return dict(
            settling_s=(float((settled[0] + 1) / 120) if len(settled) else None),
            travel_max_mm=float(max(distance)),
            final_mm=float(distance[-1]),
            rotation_max_deg=float(np.rad2deg(max(rotation))),
            final_speed_rad_s=phases[phase]["final_speed_rad_s"],
        )

    stops = {}
    for event in report["events"]:
        name = event["case"]
        stops[name] = {
            k: motion(name + "_stopped_" + k, event["stop_anchor"]) for k in ("loaded", "removed")
        }
    hold = motion("hold_loaded", report["hold_anchor"])
    removed = np.flatnonzero(data["phase"] == "loaded_stopped_removed")[-60:]
    metrics = dict(
        source_completed=report["completed"],
        qualification=False,
        removal_non_gravity_effort_max_nm=float(
            np.max(np.abs(data[applied_key][removed] - data["feedforward"][removed]))
        ),
        phases=phases,
        stops=stops,
        hold=hold,
        total_samples=len(data["q"]),
        available=int(sum(data["rejection"] == "")),
        criteria=dict(
            hold_tracking=phases["hold_loaded"]["steady_tracking_max_rad"]
            <= criteria["steady_joint_error_rad"],
            hold_travel=hold["travel_max_mm"] <= criteria["hold_travel_mm"],
            hold_rotation=hold["rotation_max_deg"] <= criteria["rotation_deg"],
            stopped_travel=all(
                p["travel_max_mm"] <= criteria["stop_travel_mm"]
                for v in stops.values()
                for p in v.values()
            ),
            stopped_rotation=all(
                p["rotation_max_deg"] <= criteria["rotation_deg"]
                for v in stops.values()
                for p in v.values()
            ),
            stopped_speed=all(
                v["loaded"]["final_speed_rad_s"] <= criteria["final_speed_rad_s"]
                for v in stops.values()
            ),
            removal_return=all(
                v["removed"]["final_mm"] <= criteria["removal_final_mm"] for v in stops.values()
            ),
            removal_speed=all(
                v["removed"]["final_speed_rad_s"] <= criteria["final_speed_rad_s"]
                for v in stops.values()
            ),
        ),
    )
    (path / "analysis.json").write_text(json.dumps(metrics, indent=2, allow_nan=False) + "\n")
    print(path.name, json.dumps({k: v for k, v in metrics.items() if k != "phases"}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", type=Path)
    for path in parser.parse_args().paths:
        analyze(path)
