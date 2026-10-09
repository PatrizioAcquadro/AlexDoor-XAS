"""Chronological frozen-trajectory test of one temporal pending-promotion criterion.

Run with Isaac Python: -m alexdoor_xas.perception.diagnostics.renewal_motion OUTPUT.
No model inference, camera extrinsics or truth enter the decision functions.
"""

import argparse
import json
import sys
import time
from collections import Counter, OrderedDict
from pathlib import Path

import h5py
import numpy as np

from alexdoor_xas.perception.evaluation import json_safe, quantiles, write_json
from alexdoor_xas.perception.point2pose.renewal_motion import (
    background_camera_motion,
    lift,
    motion_evidence,
    observed_rigid_motion,
    spatial_support,
    transform,
)


def read(path):
    return [json.loads(line) for line in path.open()]


def evaluate_condition(repo, condition, output):
    saved = repo / "outputs/evidence/b1/perception/persistent-sam3-20261008"
    attempt = saved / f"full-selected-{condition}"
    frames = read(attempt / "frames.jsonl")
    native_frames = {f["frame"]: f for f in frames if f["objects"]}
    seed_row = next(iter(native_frames.values()))["row"]
    # Evaluator-only material labels; never passed to the motion functions.
    sys.path.insert(0, str(repo / "outputs/evidence/b1/perception/p2p-sam3-followup-01"))
    from hull_evaluator import hulls, inside

    from alexdoor_xas.perception.diagnostics.replay import truth_panel

    episode = repo / (
        "datasets/recordings/b1/perception/engineering-v2/animated-door-1-88abf40/"
        f"{condition}/episode.hdf5"
    )
    pending, births, baseline_valid, accepted = {}, {}, set(), set()
    captures, max_cache, max_pending, max_anchor_bytes = OrderedDict(), 0, 0, 0
    decisions, checks, timing, supports = [], Counter(), [], []
    candidate_geometry = []
    started = time.perf_counter()
    with h5py.File(episode) as h5, (output / f"observations-{condition}.jsonl").open("x") as trace:
        observed = h5["observations"]
        dt = float(np.median(np.diff(observed["time_s"][: seed_row + 1])))
        deadline = 15 * dt
        seed_truth = truth_panel(h5, seed_row)

        def capture(row, mask_file):
            if row not in captures:
                depth = np.asarray(observed["depth_m"][row, ..., 0], float)
                depth[~np.asarray(observed["valid_depth"][row, ..., 0], bool)] = 0
                with np.load(attempt / "native" / mask_file) as masks:
                    mask = np.any(masks["masks"], axis=0)
                captures[row] = (
                    np.asarray(observed["rgb"][row]),
                    depth,
                    mask,
                    np.asarray(observed["intrinsics"][row], float),
                )
            return captures[row]

        def classify(points, row):
            # Only this evaluation branch accesses camera extrinsics and panel truth.
            world = transform(np.asarray(observed["camera_world"][row]), points)
            fixed = inside(world, hulls["Frame"])
            motion = truth_panel(h5, row) @ np.linalg.inv(seed_truth)
            moving = inside(
                transform(np.linalg.inv(motion), world), hulls["Panel"] + hulls["Handle"]
            )
            return np.where(
                fixed & moving,
                "overlap",
                np.where(fixed, "fixed-only", np.where(moving, "moving-only", "unknown")),
            )

        for line in (attempt / "native/registration.jsonl").open():
            frame = json.loads(line)
            record = native_frames[frame["frame"]]
            row, now, obj = record["row"], float(frame["capture_s"]), frame["objects"][0]
            assert now == record["time_s"]
            current_capture = capture(row, frame["mask_file"])
            ids = np.asarray(obj["map_track_indices"], int)
            valid_ids = set(ids[np.asarray(obj["map_valid"], bool)].tolist())
            xyz = lift(current_capture[1], obj["map_tracks"], current_capture[3])
            points = dict(zip(ids.tolist(), xyz, strict=True))
            raw_ids = np.asarray(obj.get("all_track_indices", []), int)
            extraction = obj.get("extraction_stats", {})
            raw_good = np.asarray(obj.get("all_visibles", []), bool)
            if len(raw_ids):
                raw_good &= np.asarray(extraction["extract_val_obj_mask"], bool)
                raw_good &= np.asarray(obj["all_uncertainties"], float) < 0.3
                raw_good &= np.asarray(extraction["extract_inside_mask"], bool)
            good_ids = set(raw_ids[raw_good].tolist())
            # Selected f2m frontend supplies rel_poses=None: native pose-stable gate
            # passes. Do not invent a tighter consecutive-pose gate in this evaluator.
            native_checks = {int(p["track_id"]): p for p in frame.get("promotion_checks", [])}
            motions = {}
            for tid, meta in list(pending.items()):
                elapsed = now - meta["birth_s"]
                if tid in good_ids:
                    meta["good"] += 1
                native = native_checks.get(tid)
                if native is not None:
                    checks["native_eligible_checks"] += 1
                    checks["good_count_mismatches"] += meta["good"] != native["metadata"]["good"]
                    meta["baseline_promotion_s"] = now if native["promoted"] else None
                if meta["good"] < 3 and elapsed <= deadline + 1e-9:
                    continue
                t0 = time.perf_counter()
                anchor_row = meta["birth_row"]
                if anchor_row not in motions:
                    source, target = [], []
                    # Frozen established set predates the whole candidate batch.
                    # Neither pending points nor candidate-derived published poses enter this fit.
                    for key, value in meta["body_points"].items():
                        if key in good_ids and key in points:
                            source.append(value)
                            target.append(points[key])
                    body, body_stats = observed_rigid_motion(
                        np.asarray(source).reshape(-1, 3), np.asarray(target).reshape(-1, 3)
                    )
                    camera, camera_stats = background_camera_motion(
                        captures[anchor_row], current_capture
                    )
                    if obj["lost"] or not record["objects"][0]["integration_accepted"]:
                        body = None
                        body_stats = dict(body_stats, reason="native_or_integration_unreliable")
                    motions[anchor_row] = body, body_stats, camera, camera_stats
                body, body_stats, camera, camera_stats = motions[anchor_row]
                z = float(meta["point"][2])
                pixel = z / min(current_capture[3][0, 0], current_capture[3][1, 1])
                evidence = motion_evidence(
                    meta["point"],
                    points.get(tid, np.full(3, np.nan)),
                    body,
                    camera,
                    elapsed_s=elapsed,
                    deadline_s=deadline + 1e-9,
                    pixel_size_m=pixel,
                    body_rms_m=body_stats.get("rms_m", 0),
                    camera_rms_m=camera_stats.get("rms_m", 0),
                )
                cost = time.perf_counter() - t0
                timing.append(cost)
                event = dict(
                    row=row,
                    capture_s=now,
                    track_id=tid,
                    **evidence,
                    body_support=body_stats,
                    camera_support=camera_stats,
                    cost_s=cost,
                )
                trace.write(json.dumps(json_safe(event), allow_nan=False) + "\n")
                checks["decisions_evaluated"] += 1
                checks[evidence["reason"]] += 1
                terminal = evidence["decision"] != "insufficient" or evidence["expired"]
                if terminal:
                    if evidence["decision"] == "body":
                        accepted.add(tid)
                    result = dict(meta, **event)
                    result.pop("body_points")
                    result["final_point"] = points.get(tid)
                    decisions.append(result)
                    del pending[tid]

            new = [tid for tid in ids if tid not in births]
            if row == seed_row:
                accepted.update(new)
            elif new:
                body_points = {
                    tid: point.copy()
                    for tid, point in points.items()
                    if tid in baseline_valid and tid in good_ids and np.isfinite(point).all()
                }
                candidate_geometry.append(
                    dict(row=row, capture_s=now, **spatial_support([points[tid] for tid in new]))
                )
                for tid in new:
                    pending[tid] = dict(
                        track_id=int(tid),
                        birth_row=row,
                        birth_s=now,
                        point=points[tid].copy(),
                        good=0,
                        body_points=body_points,
                        baseline_promotion_s=None,
                    )
            births.update({int(tid): dict(birth_row=row, point=points[tid].copy()) for tid in new})
            # Support projection only; saved P2P queries/trajectory are never changed.
            support_ids = valid_ids if row == seed_row else valid_ids & good_ids
            kept = [
                point for tid, point in points.items() if tid in accepted and tid in support_ids
            ]
            supports.append(
                dict(
                    row=row,
                    capture_s=now,
                    baseline=spatial_support([points[tid] for tid in support_ids]),
                    screened=spatial_support(kept),
                )
            )
            baseline_valid = valid_ids
            needed = {row} | {m["birth_row"] for m in pending.values()}
            for key in list(captures):
                if key not in needed:
                    del captures[key]
            max_cache, max_pending = max(max_cache, len(captures)), max(max_pending, len(pending))
            max_anchor_bytes = max(
                max_anchor_bytes, sum(sum(a.nbytes for a in cap) for cap in captures.values())
            )
            if frame["native_index"] % 600 == 0:
                print(
                    condition, row, "pending", len(pending), "terminal", len(decisions), flush=True
                )

        for meta in pending.values():
            result = dict(
                meta,
                decision="insufficient",
                reason="recording_end",
                capture_s=now,
                elapsed_s=now - meta["birth_s"],
                row=row,
            )
            result.pop("body_points")
            decisions.append(result)
        # Labels and retrospective utility annotate completed decisions only.
        for d in decisions:
            d["membership"] = str(classify(d["point"][None], d["birth_row"])[0])
            d["promotion_delay_s"] = d["elapsed_s"] if d["decision"] == "body" else None
        for birth in births.values():
            birth["membership"] = str(classify(birth["point"][None], birth["birth_row"])[0])

    write_json(output / f"decisions-{condition}.json", decisions)
    write_json(output / f"support-{condition}.json", supports)
    write_json(output / f"candidate-support-{condition}.json", candidate_geometry)
    counts = Counter((d["membership"], d["decision"]) for d in decisions)
    summary = dict(
        baseline=str(attempt.relative_to(repo)),
        seed_row=seed_row,
        seed_s=frames[seed_row]["time_s"],
        observations=len(native_frames),
        candidates=len(decisions),
        seed_references=30,
        pending_ttl=15,
        acquisition_interval_s=dt,
        deadline_s=deadline,
        outcomes={
            label: {
                decision: counts[label, decision] for decision in ("body", "static", "insufficient")
            }
            for label in ("moving-only", "fixed-only", "overlap", "unknown")
        },
        checks=dict(checks),
        max_pending=max_pending,
        max_cached_captures=max_cache,
        max_anchor_rgbd_bytes=max_anchor_bytes,
        evaluation_wall_s=time.perf_counter() - started,
        added_decision_cost_s=quantiles(np.asarray(timing)),
        promotion_delay_s=quantiles(
            np.asarray([d["elapsed_s"] for d in decisions if d["decision"] == "body"])
        ),
        support_projection={
            name: {
                key: quantiles(np.asarray([s[name][key] for s in supports]))
                for key in ("count", "second_rms_m")
            }
            for name in ("baseline", "screened")
        },
        scientific_limit=(
            "Frozen saved tracks only; projected support is not a new tracking trajectory "
            "or runtime cost."
        ),
    )
    write_json(output / f"summary-{condition}.json", summary)
    print(json.dumps(summary, allow_nan=False), flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--condition", choices=("light", "nominal"))
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[4]
    args.output.mkdir(parents=True, exist_ok=True)
    conditions = (args.condition,) if args.condition else ("light", "nominal")
    result = {
        condition: evaluate_condition(repo, condition, args.output) for condition in conditions
    }
    write_json(args.output / "summary.json", result)


if __name__ == "__main__":
    main()
