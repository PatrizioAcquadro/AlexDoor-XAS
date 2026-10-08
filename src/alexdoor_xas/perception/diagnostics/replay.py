"""Chronological two-pilot diagnostic scoring, isolated from observed-only inference."""

import json
import subprocess
from pathlib import Path

import h5py
import numpy as np
from scipy.spatial.transform import Rotation

from alexdoor_xas.action.frames import rot_z
from alexdoor_xas.perception.diagnostics.scan import PILOTS
from alexdoor_xas.perception.diagnostics.smoke import sensor_at
from alexdoor_xas.perception.evaluation import quantiles, write_json
from alexdoor_xas.perception.geometry import project
from alexdoor_xas.perception.material_zone import pose_matrix
from alexdoor_xas.perception.point2pose.runtime import tracking_provider


def truth_panel(h5, row):
    """Evaluator only; this matrix never enters the provider or worker."""
    result = np.eye(4)
    result[:3, 3] = h5["annotations/hinge_origin"][row]
    result[:3, :3] = h5["annotations/hinge_rotation"][row] @ rot_z(
        float(h5["annotations/signed_angle"][row])
    )
    return result


def visible_references(points, sensor, tolerance):
    uv, z = project(points, sensor["intrinsics"], sensor["camera_world"])
    uv = np.rint(uv).astype(int)
    h, w = sensor["rgb"].shape[:2]
    good = (z > 0) & (uv >= 0).all(1) & (uv[:, 0] < w) & (uv[:, 1] < h)
    rows = np.flatnonzero(good)
    depth = sensor["depth_m"][uv[rows, 1], uv[rows, 0], 0]
    valid = sensor["valid_depth"][uv[rows, 1], uv[rows, 0], 0]
    good[rows] &= valid & (abs(depth - z[rows]) <= 2 * tolerance)
    if good.sum() < 5:
        return False
    spread = np.linalg.svd(points[good] - points[good].mean(0), compute_uv=False)
    return bool(spread[1] / np.sqrt(good.sum()) > 2 * tolerance)


def pose_error(actual, expected):
    return [
        float(np.linalg.norm(actual[:3, 3] - expected[:3, 3])),
        float(np.rad2deg(Rotation.from_matrix(actual[:3, :3] @ expected[:3, :3].T).magnitude())),
    ]


def support_timeline(frames):
    """Sampled support runs; never infer support across absent/rejected observations."""
    intervals, active = [], None
    for row in frames:
        if row["available"]:
            if active is None:
                active = dict(first_s=row["time_s"], last_s=row["time_s"], samples=0)
            active["last_s"], active["samples"] = row["time_s"], active["samples"] + 1
        elif active is not None:
            intervals.append(active)
            active = None
    if active is not None:
        intervals.append(active)
    return dict(
        intervals=intervals,
        recoveries=max(0, len(intervals) - 1),
        longest_sampled_span_s=max((i["last_s"] - i["first_s"] for i in intervals), default=0.0),
    )


def replay_episode(path, recipe, output, models, tool_fk, distal_faces, *, stride=3):
    output.mkdir(parents=True, exist_ok=False)
    with h5py.File(path, "r") as h5:
        calibration = json.loads(h5["metadata"].attrs["calibration"])
        provider = tracking_provider(
            recipe,
            models,
            calibration,
            output,
            replay=True,
            tool_fk=tool_fk,
            distal_faces=distal_faces,
        )
        frames, events, reference, last_result_frame = [], [], None, None
        try:
            times = h5["observations/time_s"][:]
            for row in range(0, len(times), stride):
                sensor = sensor_at(h5, row)
                estimate = provider.update(sensor)
                tracker, t = provider.tracking, float(sensor["time_s"])
                if tracker.candidates and reference is None:
                    candidate = tracker.candidates[0]
                    tracker.select_zone(candidate.candidate_id, t, source="diagnostic")
                    seed_row = int(np.searchsorted(times, candidate.geometry.acquired_s))
                    reference = (
                        truth_panel(h5, seed_row),
                        candidate.initial_world.copy(),
                        candidate.surface.points[
                            :: max(1, len(candidate.surface.points) // 32)
                        ].copy(),
                    )
                    write_json(
                        output / "selection.json",
                        dict(
                            source="diagnostic",
                            candidate_id=candidate.candidate_id,
                            criterion="first automatic candidate; no truth-based reselection",
                            seed_row=seed_row,
                            selected_s=t,
                            candidates=len(tracker.candidates),
                        ),
                    )
                result = tracker.last_result
                if result is not None and result["frame"] != last_result_frame:
                    last_result_frame = result["frame"]
                    error = None
                    if reference is not None and tracker.candidates[0].state == "tracked":
                        raw_row = int(np.searchsorted(times, result["capture_s"]))
                        initial_truth, initial_zone, _ = reference
                        expected = (
                            truth_panel(h5, raw_row) @ np.linalg.inv(initial_truth) @ initial_zone
                        )
                        error = pose_error(tracker.candidates[0].pose, expected)
                    events.append(
                        dict(
                            frame=result["frame"],
                            capture_s=result["capture_s"],
                            available_s=tracker.diagnostics["available_s"],
                            latency_s=tracker.diagnostics["available_s"] - result["capture_s"],
                            inference_latency_s=result["latency_s"],
                            adapter_latency_s=result["adapter_latency_s"],
                            capture_error=error,
                            gpu_resident_bytes=result.get("gpu_resident_bytes"),
                            gpu_sampled_peak_bytes=result.get("gpu_sampled_peak_bytes"),
                            torch_peak_bytes=result["torch_peak_bytes"],
                            tsdf_rebuilds=result["tsdf_rebuilds"],
                            objects=[
                                dict(
                                    object_id=o["object_id"],
                                    lost=o["lost"],
                                    tsdf_gpu=o["tsdf_gpu"],
                                    tsdf_voxels=o["tsdf_voxels"],
                                )
                                for o in result["objects"]
                            ],
                        )
                    )
                observable = available = accurate = False
                error = reason = None
                if reference is not None:
                    initial_truth, initial_zone, points = reference
                    motion = truth_panel(h5, row) @ np.linalg.inv(initial_truth)
                    expected = motion @ initial_zone
                    visible = points @ motion[:3, :3].T + motion[:3, 3]
                    observable = visible_references(
                        visible, sensor, recipe.config["plane_tolerance_m"]
                    )
                    patch = (
                        next(
                            (
                                p
                                for p in estimate.local.patches
                                if p.patch_id == tracker.selection.patch_id
                            ),
                            None,
                        )
                        if estimate.local
                        else None
                    )
                    available = patch is not None and patch.world_pose is not None
                    reason = patch.reason if patch else estimate.reason
                    if available:
                        error = pose_error(pose_matrix(patch.world_pose), expected)
                        accurate = error[0] <= 0.01 and error[1] <= 5.0
                frames.append(
                    dict(
                        row=row,
                        time_s=t,
                        observable=observable,
                        available=available,
                        accurate=accurate,
                        current_error=error,
                        reason=reason,
                        candidates=len(tracker.candidates),
                    )
                )
                if len(frames) % 100 == 0:
                    write_json(
                        output / "status.json",
                        dict(row=row, total_rows=len(times), tracking=tracker.diagnostics),
                    )
            observed = [f for f in frames if f["observable"]]
            fraction = sum(f["available"] for f in observed) / len(observed) if observed else 0.0
            errors = np.asarray([f["current_error"] for f in observed if f["available"]]).reshape(
                -1, 2
            )
            capture_errors = np.asarray(
                [e["capture_error"] for e in events if e["capture_error"] is not None]
            ).reshape(-1, 2)
            p95 = np.quantile(errors, 0.95, axis=0) if len(errors) else None
            passed = bool(
                len(observed)
                and fraction >= 0.95
                and p95 is not None
                and p95[0] <= 0.01
                and p95[1] <= 5.0
            )
            report = dict(
                source=str(path),
                recorded_rows=len(times),
                input_rows=len(frames),
                stride=stride,
                observable_rows=len(observed),
                useful_rows=sum(f["available"] for f in observed),
                useful_availability=fraction,
                accuracy_success_rows=sum(f["available"] and f["accurate"] for f in observed),
                conditional_position_error_m=quantiles(errors[:, 0]),
                conditional_rotation_error_deg=quantiles(errors[:, 1]),
                capture_position_error_m=quantiles(capture_errors[:, 0]),
                capture_rotation_error_deg=quantiles(capture_errors[:, 1]),
                latency_s=quantiles(np.asarray([e["latency_s"] for e in events])),
                diagnostic_passed=passed,
                complete=True,
                denominator="all sampled rows retained; independent projected RGB-D references",
                official_qualification_evaluated=False,
                qualified=False,
                offline_passed=False,
                dynamic_passed=False,
                loaded_contact_admitted=False,
                training_started=False,
                sealed_test_evaluated=False,
                sampled_support=support_timeline(frames),
                missing_bounds=[
                    "hardware calibration/FK",
                    "temporal motion/stop",
                    "loaded response",
                ],
                last_tracking_state=tracker.diagnostics,
            )
            write_json(output / "report.json", report)
            return report
        except Exception as error:
            write_json(
                output / "failure.json",
                dict(
                    error=f"{type(error).__name__}: {error}",
                    completed_rows=len(frames),
                    qualified=False,
                ),
            )
            raise
        finally:
            write_json(output / "frames.json", frames)
            write_json(output / "events.json", events)
            provider.close()


def run_replay(paths, recipe, output, models, *, stride=3):
    expected = {(pilot, condition) for pilot in PILOTS for condition in ("nominal", "light")}
    if {(p.parent.parent.name, p.parent.name) for p in paths} != expected or len(paths) != 4:
        raise ValueError("Exactly the four authorized pilot recordings are required")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    from ihmc_alex_isaaclab._paths import REPOSITORY_ROOT as alex_root

    from alexdoor_xas.assets.purdue import derive_push_geometry
    from alexdoor_xas.policies.purdue import PurdueFK

    urdf = (
        alex_root
        / "assets/robots/alex_purdue/urdf/baseline/alex_purdue_wsg32_umi_v1_full_convex.urdf"
    )
    tool_fk = PurdueFK(urdf)
    distal_faces = derive_push_geometry(urdf).contact_covers()
    write_json(
        output / "protocol.json",
        dict(
            mode="point2pose-pilot-replay",
            source_commit=subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[4], text=True
            ).strip(),
            episodes=[str(p) for p in paths],
            stride=stride,
            useful_availability_min=0.95,
            p95_position_limit_m=0.01,
            p95_rotation_limit_deg=5.0,
            max_dynamic_age_s=0.15,
            recipe=recipe.to_dict(),
            annotations="evaluator only",
            loaded_contact_admitted=False,
            official_qualification_evaluated=False,
        ),
    )
    reports = []
    for path in paths:
        reports.append(
            replay_episode(
                path,
                recipe,
                output / path.parent.parent.name / path.parent.name,
                models,
                tool_fk,
                distal_faces,
                stride=stride,
            )
        )
        write_json(
            output / "report.json",
            dict(
                episodes=reports,
                qualified=False,
                diagnostic_passed=all(r["diagnostic_passed"] for r in reports),
            ),
        )
    return reports
