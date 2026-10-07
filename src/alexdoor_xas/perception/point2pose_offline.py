"""Serial acquisition-time evaluation; no operational scheduler or contact publication."""

import json
import subprocess
import time
from collections import Counter
from pathlib import Path

import h5py
import numpy as np

from alexdoor_xas.perception.evaluation import json_safe, quantiles, write_json
from alexdoor_xas.perception.geometry import surfaces
from alexdoor_xas.perception.material_zone import transported_zone
from alexdoor_xas.perception.panel_tracking import PanelTracking, measured_registration
from alexdoor_xas.perception.point2pose_diagnostics import sensor_at
from alexdoor_xas.perception.point2pose_replay import pose_error, truth_panel, visible_references
from alexdoor_xas.perception.point2pose_runtime import Point2PoseWorker
from alexdoor_xas.perception.point2pose_worker import unpack_array
from alexdoor_xas.perception.provider import ModelWorker
from alexdoor_xas.perception.scan_diagnostics import PILOTS


class SerialPoint2PoseEngine:
    """Synchronous native requests, also used by the existing candidate initializer."""

    def __init__(self, factory):
        self.factory = factory
        self.worker = self.result = None

    def reset(self):
        self.close()
        self.result = None

    def prepare(self):
        self.worker = self.factory()
        self.worker.first_request = False  # Startup is measured before the first acquisition.

    def submit(self, sensor, masks=None, *, start_s=None):
        self.result = self.worker.infer(sensor, masks)
        if self.result["frame"] != int(sensor["frame"]) or self.result["capture_s"] != float(
            sensor["time_s"]
        ):
            raise ValueError("offline_result_capture_mismatch")
        return True

    def close(self):
        if self.worker is not None:
            self.worker.close()
            self.worker = None


def finite_pose(pose):
    pose = np.asarray(pose)
    return bool(
        pose.shape == (4, 4)
        and np.isfinite(pose).all()
        and np.allclose(pose[3], [0, 0, 0, 1], atol=1e-6, rtol=0)
        and np.allclose(pose[:3, :3].T @ pose[:3, :3], np.eye(3), atol=1e-4, rtol=0)
        and abs(np.linalg.det(pose[:3, :3]) - 1) <= 1e-4
    )


def score_objects(candidates, sensor, result, motion, config, checks):
    """Truth motion is evaluator-only and cannot affect the next native request."""
    if [obj["object_id"] for obj in result["objects"]] != list(range(len(candidates))):
        raise ValueError("offline_native_identity_changed")
    records = []
    for candidate, raw in zip(candidates, result["objects"], strict=True):
        obj = {
            key: unpack_array(value) if isinstance(value, dict) and "data" in value else value
            for key, value in raw.items()
        }
        pose_valid = finite_pose(obj["camera_from_map"])
        pose = (
            transported_zone(
                sensor["camera_world"],
                obj["camera_from_map"],
                candidate.map_from_object,
                candidate.object_from_zone,
            )
            if pose_valid
            else None
        )
        points = candidate.surface.points[:: max(1, len(candidate.surface.points) // 32)]
        observable = visible_references(
            points @ motion[:3, :3].T + motion[:3, 3], sensor, config["plane_tolerance_m"]
        )
        error = None if pose is None else pose_error(pose, motion @ candidate.initial_world)
        native_tracking = pose_valid and not obj["lost"]
        p = a = None
        pairs = 0
        if pose_valid:
            source, p, a = measured_registration(obj, config["plane_tolerance_m"])
            pairs = len(source)
        check = checks[obj["object_id"]]
        reasons = []
        if not check["accepted"]:
            reasons.append(check["reason"])
        if not pose_valid:
            reasons.append("invalid_native_pose")
        if obj["lost"]:
            reasons.append(
                "native_tracking_lost" if obj.get("registration_evaluated", True)
                else "registration_deferred"
            )
        if p is None or a is None or not np.isfinite([p, a]).all():
            reasons.append(
                "insufficient_measured_pairs" if pairs < 5 else "degenerate_registration"
            )
        records.append(
            dict(
                candidate_id=candidate.candidate_id,
                object_id=obj["object_id"],
                camera_from_map=obj["camera_from_map"],
                world_pose=pose,
                native_lost=bool(obj["lost"]),
                registration_evaluated=obj.get("registration_evaluated", True),
                native_pose_valid=pose_valid,
                native_tracking=bool(native_tracking),
                integration_accepted=not reasons,
                integration_rejections=reasons,
                initialization_accepted=check["accepted"],
                observable=observable,
                capture_error=error,
                accurate=bool(native_tracking and error[0] <= 0.01 and error[1] <= 5),
                measured_pairs=pairs,
                registration_position_m=p,
                registration_rotation_rad=a,
                tsdf_gpu=obj["tsdf_gpu"],
                tsdf_voxels=obj["tsdf_voxels"],
            )
        )
    return records


def continuity(rows, field):
    """Runs and transitions in acquisition time; no support across missing observations."""
    runs, gaps = [], []
    active = None
    gap = None
    for row in rows:
        if row.get(field, False):
            if active is None:
                if gap is not None:
                    gap.update(recovered_s=row["time_s"], duration_s=row["time_s"] - gap["lost_s"])
                    gaps.append(gap)
                    gap = None
                active = dict(first_s=row["time_s"], last_s=row["time_s"], samples=0)
            active.update(last_s=row["time_s"], samples=active["samples"] + 1)
        elif active is not None:
            runs.append(active)
            gap = dict(lost_s=row["time_s"], recovered_s=None, duration_s=None)
            active = None
    if active is not None:
        runs.append(active)
    if gap is not None:
        gap["observed_until_s"] = rows[-1]["time_s"]
        gap["unrecovered_observed_span_s"] = rows[-1]["time_s"] - gap["lost_s"]
        gaps.append(gap)
    return dict(
        intervals=runs,
        interruptions=gaps,
        losses=len(gaps),
        recoveries=sum(g["recovered_s"] is not None for g in gaps),
        longest_sampled_span_s=max((r["last_s"] - r["first_s"] for r in runs), default=0.0),
    )


def summarize_rows(rows):
    groups = dict(
        native_all_finite=lambda r: r.get("native_pose_valid", False),
        native_tracking=lambda r: r.get("native_tracking", False),
        integration_accepted=lambda r: r.get("integration_accepted", False),
        integration_rejected=lambda r: (
            r.get("native_pose_valid", False) and not r.get("integration_accepted", False)
        ),
        native_lost_finite=lambda r: (
            r.get("native_pose_valid", False) and r.get("native_lost", False)
        ),
    )
    errors = {}
    for name, include in groups.items():
        values = np.asarray(
            [r["capture_error"] for r in rows if include(r) and not r["is_seed"]]
        ).reshape(-1, 2)
        errors[name] = dict(
            position_m=quantiles(values[:, 0]), rotation_deg=quantiles(values[:, 1])
        )
    n = len(rows)
    return dict(
        rows=n,
        native_tracking_rows=sum(r.get("native_tracking", False) for r in rows),
        integration_accepted_rows=sum(r.get("integration_accepted", False) for r in rows),
        native_coverage=sum(r.get("native_tracking", False) for r in rows) / n if n else 0.0,
        integration_coverage=sum(r.get("integration_accepted", False) for r in rows) / n
        if n
        else 0.0,
        accurate_native_coverage=sum(r.get("accurate", False) for r in rows) / n if n else 0.0,
        observable_rows=sum(r.get("observable") is True for r in rows),
        unknown_observability_rows=sum(r.get("observable") is None for r in rows),
        native_lost_rows=sum(r.get("native_lost", False) for r in rows),
        invalid_native_pose_rows=sum(
            "native_pose_valid" in r and not r["native_pose_valid"] for r in rows
        ),
        rejection_causes=dict(
            Counter(reason for r in rows for reason in r.get("integration_rejections", []))
        ),
        errors_non_seed=errors,
    )


def summarize_candidate(frames, candidate_id, seed_row):
    rows = []
    for frame in frames:
        obj = next((o for o in frame["objects"] if o["candidate_id"] == candidate_id), {})
        rows.append(
            dict(obj, time_s=frame["time_s"], row=frame["row"], is_seed=frame["row"] == seed_row)
        )
    return dict(
        candidate_id=candidate_id,
        ownership_qualified=False,
        all_frames=summarize_rows(rows),
        after_seed=summarize_rows([r for r in rows if r["row"] > seed_row]),
        observable=summarize_rows([r for r in rows if r.get("observable") is True]),
        native_continuity=continuity(rows, "native_tracking"),
        integration_continuity=continuity(rows, "integration_accepted"),
    )


def offline_episode(
    path,
    recipe,
    output,
    models,
    *,
    initialization_only=False,
    capture_window_s=None,
    registration_diagnostics=False,
    use_key_frame_graph=True,
    allow_partial_reference_batch=False,
    refit_seed_rollback=False,
    selected_registration_only=False,
):
    """One fresh attempt, retaining terminal failures in the scheduled denominator."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    config = recipe.config
    frames, candidates, checks = [], [], []
    engine = cues = tracker = None
    failure = seed_row = seed_truth = None
    interrupted = False
    startup = {}
    inference_wall_s = 0.0
    first_supported = {}
    first_native = {}
    initialization = dict(native_completed=False, integration_checks=[], seed_row=None)
    with h5py.File(path, "r") as h5, (output / "visual.log").open("w") as log:
        times, frame_ids = h5["observations/time_s"][:], h5["observations/frame"][:]
        end_s = (
            config["inspection"]["sample_times_s"][-1] if initialization_only else float(times[-1])
        )
        first_s = float(times[0])
        if capture_window_s is not None:
            if (
                initialization_only
                or not np.isfinite(capture_window_s).all()
                or not (times[0] <= capture_window_s[0] <= capture_window_s[1] <= times[-1])
            ):
                raise ValueError("invalid_offline_capture_window")
            first_s, end_s = capture_window_s
        scheduled = np.flatnonzero((times >= first_s - 1e-9) & (times <= end_s + 1e-9))
        next_semantic = max(first_s, config["inspection"]["sample_times_s"][0])
        semantic_end_s = (
            end_s if capture_window_s is not None else config["inspection"]["sample_times_s"][-1]
        )
        calibration = json.loads(h5["metadata"].attrs["calibration"])
        try:
            if (
                not np.isfinite(times).all()
                or not (np.diff(times) > 0).all()
                or not (np.diff(frame_ids) > 0).all()
            ):
                raise ValueError("nonmonotonic_offline_recording")
            if not np.allclose(np.diff(times), 1 / 60, atol=1e-6, rtol=0):
                raise ValueError("offline_recording_requires_60_hz")
            started = time.perf_counter()
            cues = ModelWorker(models, config, log)
            startup["visual_s"] = time.perf_counter() - started
            engine = SerialPoint2PoseEngine(
                lambda: Point2PoseWorker(
                    Path(models) / "point2pose",
                    calibration["depth_interval_m"],
                    config["plane_tolerance_m"],
                    1,
                    output / "native",
                    diagnostic_only=True,
                    **({"registration_diagnostics": True} if registration_diagnostics else {}),
                    **({"use_key_frame_graph": False} if not use_key_frame_graph else {}),
                    **(
                        {"allow_partial_reference_batch": True}
                        if allow_partial_reference_batch
                        else {}
                    ),
                    **({"refit_seed_rollback": True} if refit_seed_rollback else {}),
                    **({"selected_registration_only": True} if selected_registration_only else {}),
                )
            )
            tracker = PanelTracking(engine, config)
            tracker.before_start = cues.close
            engine.prepare()
            startup["point2pose_s"] = engine.worker.boot_latency_s
        except KeyboardInterrupt:
            interrupted = True
            failure = "user_requested_stop"
        except Exception as error:
            failure = f"{type(error).__name__}: {error}"
        write_json(output / "startup.json", startup)
        try:
            with (output / "frames.jsonl").open("w", buffering=1) as stream:
                for row in scheduled:
                    row, t = int(row), float(times[row])
                    if t > end_s + 1e-9:
                        break
                    record = dict(
                        row=row,
                        frame=int(frame_ids[row]),
                        time_s=float(t),
                        status="not_initialized",
                        is_seed=False,
                        objects=[],
                        latency=None,
                    )
                    result = None
                    sensor = sensor_at(h5, row)
                    request_started = time.perf_counter()
                    if failure is not None:
                        record["status"] = "not_processed"
                    else:
                        try:
                            if (
                                not candidates
                                and next_semantic
                                <= t
                                <= semantic_end_s + 1e-9
                            ):
                                cue = cues.infer(sensor["rgb"])
                                next_semantic = float(t) + config["semantic_period_s"]
                                tracker.initialize(surfaces(cue, sensor, config), sensor)
                                record["semantic_latency_s"] = cue["latency_s"]
                            elif candidates:
                                engine.submit(sensor)
                            result = engine.result
                            if result is not None:
                                record["status"] = "native_result"
                                record["latency"] = dict(
                                    request_s=time.perf_counter() - request_started,
                                    native_ipc_s=result["latency_s"],
                                    native_compute_s=result.get("model_latency_s"),
                                    diagnostic_export_s=result.get("diagnostic_export_s"),
                                )
                                record["registration_schedule"] = result.get(
                                    "registration_schedule"
                                )
                                record["candidate_history"] = result.get("candidate_history")
                        except Exception as error:
                            failure = f"{type(error).__name__}: {error}"
                            record.update(status="process_error", error=failure)
                        finally:
                            if tracker.candidates and seed_row is None:
                                candidates, seed_row = tracker.candidates, row
                                seed_truth = truth_panel(h5, row)
                                initialization["seed_row"] = row
                                initialization["seed_time_s"] = float(t)
                                if initialization_only:
                                    end_s = float(t) + 1.0
                                write_json(
                                    output / "selection.json",
                                    dict(
                                        primary_candidate=candidates[0].candidate_id,
                                        seed_row=row,
                                        seed_frame=int(frame_ids[row]),
                                        seed_time_s=float(t),
                                        criterion="first eligible automatic candidate; no truth",
                                        candidates=[c.candidate_id for c in candidates],
                                        pre_native_rejections=tracker.diagnostics.get(
                                            "initialization_rejections", []
                                        ),
                                        ownership_qualified=False,
                                    ),
                                )
                    record["is_seed"] = row == seed_row
                    if candidates:
                        motion = truth_panel(h5, row) @ np.linalg.inv(seed_truth)
                        try:
                            if result is not None:
                                if row == seed_row:
                                    checks = result["initialization_checks"]
                                    if [c["object_id"] for c in checks] != list(
                                        range(len(candidates))
                                    ):
                                        raise ValueError("offline_initialization_checks_mismatch")
                                    initialization.update(
                                        native_completed=True, integration_checks=checks
                                    )
                                record["objects"] = score_objects(
                                    candidates, sensor, result, motion, config, checks
                                )
                                inference_wall_s += record["latency"]["request_s"]
                                for obj in record["objects"]:
                                    if (
                                        obj["native_tracking"]
                                        and obj["candidate_id"] not in first_native
                                    ):
                                        first_native[obj["candidate_id"]] = dict(
                                            image_time_s=float(t),
                                            image_elapsed_from_seed_s=float(t - times[seed_row]),
                                            inference_wall_from_seed_s=inference_wall_s,
                                        )
                                    if (
                                        obj["integration_accepted"]
                                        and obj["candidate_id"] not in first_supported
                                    ):
                                        first_supported[obj["candidate_id"]] = dict(
                                            image_time_s=float(t),
                                            image_elapsed_from_seed_s=float(t - times[seed_row]),
                                            inference_wall_from_seed_s=inference_wall_s,
                                        )
                                record["gpu"] = {
                                    key: result.get(key)
                                    for key in (
                                        "gpu_resident_bytes",
                                        "gpu_sampled_peak_bytes",
                                        "torch_peak_bytes",
                                        "tsdf_rebuilds",
                                    )
                                }
                            else:
                                for candidate in candidates:
                                    points = candidate.surface.points[
                                        :: max(1, len(candidate.surface.points) // 32)
                                    ]
                                    record["objects"].append(
                                        dict(
                                            candidate_id=candidate.candidate_id,
                                            observable=visible_references(
                                                points @ motion[:3, :3].T + motion[:3, 3],
                                                sensor,
                                                config["plane_tolerance_m"],
                                            ),
                                        )
                                    )
                        except Exception as error:
                            failure = f"{type(error).__name__}: {error}"
                            record.update(status="process_error", error=failure)
                            record["latency"] = dict(
                                request_s=time.perf_counter() - request_started,
                                native_ipc_s=None,
                                native_compute_s=None,
                            )
                    frames.append(record)
                    stream.write(json.dumps(json_safe(record), allow_nan=False) + "\n")
                    if row % 100 == 0 or row == scheduled[-1]:
                        status = dict(
                            row=row,
                            recorded_rows=len(times),
                            time_s=float(t),
                            seed_row=seed_row,
                            failure=failure,
                            initialization_only=initialization_only,
                        )
                        write_json(output / "status.json", status)
                        print(
                            f"{output}: row {row}/{len(times) - 1}, {record['status']}", flush=True
                        )
        except KeyboardInterrupt:
            interrupted = True
            failure = "user_requested_stop"
            # Keep completed lines intact. The in-flight request is unavailable;
            # never infer its result, observability or latency from a later response.
            with (output / "frames.jsonl").open("a", buffering=1) as stream:
                for row in scheduled[len(frames) :]:
                    row = int(row)
                    if times[row] > end_s + 1e-9:
                        break
                    record = dict(
                        row=row,
                        frame=int(frame_ids[row]),
                        time_s=float(times[row]),
                        status="not_processed",
                        reason="user_requested_stop",
                        is_seed=False,
                        objects=[],
                        latency=None,
                    )
                    frames.append(record)
                    stream.write(json.dumps(record, allow_nan=False) + "\n")
        finally:
            if engine is not None:
                engine.close()
            if cues is not None and cues.process.poll() is None:
                cues.close()
        if failure is not None:
            write_json(output / "failure.json", dict(error=failure, qualified=False))
        initialization.update(
            first_supported=first_supported,
            first_native_tracking=first_native,
            reason="no_eligible_automatic_candidate" if seed_row is None else failure,
            pre_native_rejections=tracker.diagnostics.get("initialization_rejections", [])
            if tracker
            else [],
            failure=failure,
        )
        write_json(output / "initialization.json", initialization)
        summaries = [summarize_candidate(frames, c.candidate_id, seed_row) for c in candidates]
        if not summaries:
            summaries = [dict(candidate_id=None, all_frames=summarize_rows(frames))]
        report = dict(
            source=str(path),
            initialization_only=initialization_only,
            capture_window_s=capture_window_s,
            registration_diagnostics=registration_diagnostics,
            recorded_rows=len(times),
            scheduled_rows=len(frames),
            inspected_rows=sum(r["status"] != "not_processed" for r in frames),
            native_result_rows=sum(r["status"] == "native_result" for r in frames),
            not_processed_rows=sum(r["status"] == "not_processed" for r in frames),
            statuses=dict(Counter(r["status"] for r in frames)),
            cadence_hz=60,
            stride=1,
            capture_interval_s=quantiles(np.diff(times)),
            complete=failure is None,
            termination="user_requested_stop" if interrupted else None,
            failure=failure,
            initialization=initialization,
            primary=summaries[0],
            candidates=summaries,
            latency={
                key: quantiles(
                    np.asarray(
                        [
                            r["latency"][key]
                            for r in frames
                            if r["latency"] and r["latency"][key] is not None
                        ]
                    )
                )
                for key in ("request_s", "native_ipc_s", "native_compute_s")
            },
            startup=startup,
            latency_affects_outcome=False,
            position_reference_m=0.01,
            rotation_reference_deg=5,
            seed_zero_is_alignment=True,
            absolute_initial_pose_accuracy_evaluated=False,
            official_qualification_evaluated=False,
            offline_passed=False,
            dynamic_passed=False,
            qualified=False,
            loaded_contact_admitted=False,
            training_started=False,
        )
        write_json(output / "report.json", report)
        if interrupted:
            write_json(
                output / "status.json",
                dict(
                    termination="user_requested_stop",
                    recorded_rows=len(times),
                    inspected_rows=report["inspected_rows"],
                    not_processed_rows=report["not_processed_rows"],
                    workers_closed=True,
                    no_restart=True,
                ),
            )
        return report


def run_offline(paths, recipe, output, models):
    expected = {(pilot, condition) for pilot in PILOTS for condition in ("nominal", "light")}
    if len(paths) != 4 or {(p.parent.parent.name, p.parent.name) for p in paths} != expected:
        raise ValueError("Exactly the four authorized pilot recordings are required")
    recorded_rows = {}
    for path in paths:
        with h5py.File(path, "r") as h5:
            recorded_rows[str(path)] = len(h5["observations/time_s"])
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    write_json(
        output / "protocol.json",
        dict(
            mode="point2pose-offline",
            cadence_hz=60,
            stride=1,
            initialization_attempts_per_episode=3,
            extra_initialization_window_s=1.0,
            episodes=[str(p) for p in paths],
            recorded_rows=recorded_rows,
            recipe=recipe.to_dict(),
            annotations="evaluator only",
            latency_affects_outcome=False,
            source_commit=subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[3], text=True
            ).strip(),
            loaded_contact_admitted=False,
            official_qualification_evaluated=False,
        ),
    )
    reports = []
    for path in paths:
        for attempt in range(1, 4):
            reports.append(
                offline_episode(
                    path,
                    recipe,
                    output / path.parent.parent.name / path.parent.name / f"attempt-{attempt}",
                    models,
                    initialization_only=attempt > 1,
                )
            )
            write_json(
                output / "report.json",
                dict(
                    attempts=reports,
                    completed_attempts=sum(r["complete"] for r in reports),
                    finished_attempts=len(reports),
                    not_started_attempts=12 - len(reports),
                    expected_attempts=12,
                    complete=len(reports) == 12 and all(r["complete"] for r in reports),
                    full_recording_frames=sum(
                        r["scheduled_rows"] for r in reports if not r["initialization_only"]
                    ),
                    expected_full_recording_frames=sum(recorded_rows.values()),
                    not_processed_full_recording_frames=sum(recorded_rows.values())
                    - sum(r["inspected_rows"] for r in reports if not r["initialization_only"]),
                    native_initializations=sum(
                        r["initialization"]["native_completed"] for r in reports
                    ),
                    integration_initializations=sum(
                        r["initialization"]["native_completed"]
                        and all(c["accepted"] for c in r["initialization"]["integration_checks"])
                        for r in reports
                    ),
                    qualified=False,
                    offline_passed=False,
                    dynamic_passed=False,
                    loaded_contact_admitted=False,
                    termination=reports[-1].get("termination"),
                ),
            )
            if reports[-1].get("termination") == "user_requested_stop":
                return reports
    return reports
