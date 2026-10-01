"""Full-state scoring of causal outputs. Simulator annotations are read only here."""

import json
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
from PIL import Image, ImageDraw
from scipy.spatial.transform import Rotation

from alexdoor_xas.action.frames import ObjectFrame, rot_z
from alexdoor_xas.policies.observations import B1Observer
from alexdoor_xas.recording.b1 import OBS_KEYS, PHASES

LIMITS = dict(
    hinge_origin_m=0.01,
    hinge_rotation_deg=5.0,
    dimensions_m=0.01,
    contact_local_m=0.01,
    contact_rotation_local_deg=5.0,
    contact_position_m=0.01,
    contact_rotation_deg=5.0,
    panel_rotation_deg=5.0,
    signed_angle_deg=5.0,
)
COMPONENTS = (*LIMITS, "width_m", "height_m", "thickness_m")
MANIPULATION = [PHASES.index(p) for p in ("contact", "push", "hold")]


@dataclass(frozen=True)
class MaterialContactReference:
    """Evaluator-only association at selection; independent surface query remains 6.0E work."""

    selection_id: str
    selected_s: float
    leaf_local_pose: ObjectFrame
    footprint_leaf: tuple[np.ndarray, np.ndarray]

    def require_selection(self, selection):
        if (selection.selection_id, selection.selected_s) != (self.selection_id, self.selected_s):
            raise ValueError("contact_correspondence_requires_new_surface_query")


def rotation_error(a, b):
    if a is None or np.asarray(a).shape != (3, 3) or not np.isfinite(a).all():
        return np.nan
    return float(np.rad2deg(Rotation.from_matrix(np.asarray(a) @ b.T).magnitude()))


def position_error(a, b):
    if a is None or np.asarray(a).shape != np.asarray(b).shape or not np.isfinite(a).all():
        return np.nan
    return float(np.linalg.norm(np.asarray(a) - b))


def state_errors(estimate, truth):
    errors = dict.fromkeys(COMPONENTS, np.nan)
    frame = estimate.frame
    if frame is not None:
        errors["hinge_origin_m"] = position_error(frame.origin, truth["hinge_origin"])
        errors["hinge_rotation_deg"] = rotation_error(frame.rot, truth["hinge_rotation"])
    if estimate.signed_angle is not None:
        delta = estimate.signed_angle - truth["signed_angle"]
        errors["signed_angle_deg"] = float(
            abs(np.rad2deg(np.arctan2(np.sin(delta), np.cos(delta))))
        )
    true_panel = truth["hinge_rotation"] @ rot_z(truth["signed_angle"])
    errors["panel_rotation_deg"] = rotation_error(estimate.panel_rotation, true_panel)
    if estimate.dimensions is not None:
        delta = abs(estimate.dimensions - truth["dimensions"])
        if np.isfinite(delta).all():
            errors["dimensions_m"] = float(delta.max())
        errors.update(zip(("width_m", "height_m", "thickness_m"), delta, strict=True))
    errors["contact_position_m"] = position_error(
        estimate.contact_position, truth["contact_position"]
    )
    errors["contact_rotation_deg"] = rotation_error(
        estimate.contact_rotation, truth["contact_rotation"]
    )
    if frame is not None and estimate.panel_rotation is not None:
        if estimate.contact_position is not None:
            local = estimate.panel_rotation.T @ (estimate.contact_position - frame.origin)
            errors["contact_local_m"] = position_error(local, truth["contact_local"])
        if estimate.contact_rotation is not None:
            local_r = estimate.panel_rotation.T @ estimate.contact_rotation
            errors["contact_rotation_local_deg"] = rotation_error(
                local_r, truth["contact_rotation_local"]
            )
    return np.array([errors[k] for k in COMPONENTS], dtype=float)


def quantiles(values):
    values = values[np.isfinite(values)]
    if not len(values):
        return dict(n=0, p50=None, p95=None, maximum=None)
    return dict(
        n=len(values),
        p50=float(np.median(values)),
        p95=float(np.quantile(values, 0.95)),
        maximum=float(values.max()),
    )


def score(errors, accepted, reasons, selection, causes=None):
    indices = np.flatnonzero(selection)
    errors, accepted = errors[indices], accepted[indices]
    counts = Counter(str(reasons[i]) for i in indices)
    limits = np.array(list(LIMITS.values()))
    accurate = np.isfinite(errors[:, : len(limits)]).all(1) & (
        errors[:, : len(limits)] <= limits
    ).all(1)
    coverage = float(accepted.mean()) if len(indices) else 0.0
    precision = float(accurate[accepted].mean()) if accepted.any() else 0.0
    all_errors = {k: quantiles(errors[:, i]) for i, k in enumerate(COMPONENTS)}
    accepted_errors = {k: quantiles(errors[accepted, i]) for i, k in enumerate(COMPONENTS)}
    passed = (
        coverage >= 0.95
        and precision >= 0.95
        and all(
            accepted_errors[k]["p95"] is not None and accepted_errors[k]["p95"] <= limit
            for k, limit in LIMITS.items()
        )
    )
    return dict(
        observations=len(indices),
        accepted=int(accepted.sum()),
        valid_coverage=coverage,
        accepted_state_precision=precision,
        accurate_coverage=float(accurate.mean()) if len(indices) else 0,
        offline_passed=passed,
        reasons=dict(counts),
        rejection_causes=dict(
            Counter(
                cause
                for i, keep in zip(indices, accepted, strict=True)
                if not keep
                for cause in (
                    str(causes[i]).split("|") if causes is not None else [str(reasons[i])]
                )
                if cause
            )
        ),
        errors_all_finite=all_errors,
        errors_rejected={k: quantiles(errors[~accepted, i]) for i, k in enumerate(COMPONENTS)},
        errors_accepted=accepted_errors,
    )


def json_safe(value):
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, dict):
        return {k: json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [json_safe(v) for v in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (int, np.integer)):
        return int(value)
    return value


def write_json(path, value):
    Path(path).write_text(json.dumps(json_safe(value), indent=2, allow_nan=False) + "\n")


def mask_evidence(path, sensor, cue, provider):
    rgb = sensor["rgb"].copy()
    h, w = cue["shape"]
    palette = ([0, 210, 255], [255, 180, 0], [190, 80, 255], [30, 255, 90])
    for i, packed in enumerate(cue["masks"]):
        mask = (
            np.unpackbits(np.frombuffer(packed, np.uint8), count=h * w).reshape(h, w).astype(bool)
        )
        rgb[mask] = (0.65 * rgb[mask] + 0.35 * np.array(palette[i % len(palette)])).astype(np.uint8)
    from alexdoor_xas.perception.geometry import project

    picture = Image.fromarray(rgb)
    draw = ImageDraw.Draw(picture)
    geometry = {}
    for i, surface in enumerate(provider.static):
        cloud = surface.points[:: max(1, len(surface.points) // 600)]
        pixels, depth = project(cloud, sensor["intrinsics"], sensor["camera_world"])
        shown = (depth > 0) & np.isfinite(pixels).all(1)
        color = (255, 60, 60) if surface is provider.closed else (255, 235, 40)
        for x, y in pixels[shown]:
            if 0 <= x < w and 0 <= y < h:
                draw.point((int(x), int(y)), fill=color)
        geometry[f"surface_{i}"] = cloud
        geometry[f"normal_{i}"] = surface.normal
        geometry[f"extent_{i}"] = surface.extent_points
        if surface is provider.closed:
            geometry["selected_surface_index"] = np.array(i)
            geometry["selected_bounds"] = surface.bounds
    estimate = provider.last_estimate
    draw.text(
        (12, 12),
        f"Observed surfaces: red=selected, yellow=competing; {estimate.reason}",
        fill=(255, 255, 255),
        stroke_width=1,
        stroke_fill=(0, 0, 0),
    )
    if estimate.frame is not None:
        geometry["hinge_origin"] = estimate.frame.origin
    if estimate.contact_position is not None:
        geometry["contact_position"] = estimate.contact_position
    if estimate.panel_rotation is not None:
        geometry["panel_rotation"] = estimate.panel_rotation
    picture.save(path)
    np.savez_compressed(path.with_suffix(".npz"), **geometry)


def evaluate_episode(path, provider, output):
    output.mkdir(parents=True, exist_ok=False)
    observer = B1Observer(provider, provider.binding)
    with h5py.File(path, "r") as h5:
        metadata = json.loads(h5["metadata"].attrs["episode"])
        if metadata["split"] not in ("train", "development"):
            raise ValueError("Sealed test cannot enter prototype evaluation")
        # Evaluator-owned labels are never passed to the observer or provider.
        truth = {k: v[:] for k, v in h5["annotations"].items()}
        observations = h5["observations"]
        n = len(observations["time_s"])
        errors = np.full((n, len(COMPONENTS)), np.nan)
        accepted = np.zeros(n, bool)
        confidence = np.zeros(n)
        reasons = np.empty(n, dtype="U64")
        causes = np.empty(n, dtype="U256")
        elapsed = np.zeros(n)
        traces, latencies = [], []
        last_cue = None
        evidence_times = [0.0, *provider.config["inspection"]["sample_times_s"]]
        started = time.perf_counter()
        for row in range(n):
            sensor = {key: observations[key][row] for key in OBS_KEYS}
            tic = time.perf_counter()
            encoded = observer.update(sensor)
            elapsed[row] = time.perf_counter() - tic
            estimate = provider.last_estimate
            accepted[row] = encoded.valid
            confidence[row] = estimate.confidence
            reasons[row] = estimate.reason if not encoded.valid else "observed"
            causes[row] = "|".join(provider.diagnostics.get("missing", [str(reasons[row])]))
            errors[row] = state_errors(estimate, {key: value[row] for key, value in truth.items()})
            cue_frame = provider.diagnostics.get("cue_frame")
            if cue_frame is not None and cue_frame != last_cue:
                last_cue = cue_frame
                latencies.append(provider.diagnostics["latency_s"])
                trace = dict(
                    row=row,
                    frame=int(sensor["frame"]),
                    time_s=float(sensor["time_s"]),
                    accepted=bool(accepted[row]),
                    reason=str(reasons[row]),
                    diagnostics=dict(provider.diagnostics),
                )
                traces.append(trace)
                capture_time = float(provider.last_cue[0]["time_s"]) if provider.last_cue else -1
                crossed_hold = bool(evidence_times and capture_time >= evidence_times[0])
                if crossed_hold:
                    evidence_times.pop(0)
                if provider.last_cue is not None and (
                    len(traces) <= 7 or len(traces) % 100 == 0 or crossed_hold
                ):
                    capture, cue = provider.last_cue
                    mask_evidence(
                        output / f"mask-frame-{int(capture['frame']):06d}.png",
                        capture,
                        cue,
                        provider,
                    )
            if row % 600 == 0:
                write_json(
                    output / "status.json",
                    dict(
                        state="running",
                        row=row,
                        observations=n,
                        elapsed_s=time.perf_counter() - started,
                        accepted=int(accepted[: row + 1].sum()),
                        reason=str(reasons[row]),
                        diagnostics=provider.diagnostics,
                    ),
                )
                print(
                    json.dumps(
                        dict(
                            asset_id=metadata["asset_id"],
                            condition=metadata["condition"],
                            row=row,
                            observations=n,
                            reason=str(reasons[row]),
                        )
                    ),
                    flush=True,
                )
        phase = truth["phase"]
        manipulation = np.isin(phase, MANIPULATION)
        report = dict(
            asset_id=metadata["asset_id"],
            split=metadata["split"],
            handedness=metadata["handedness"],
            condition=metadata["condition"],
            path=str(path),
            observations=n,
            elapsed_s=time.perf_counter() - started,
            manipulation=score(errors, accepted, reasons, manipulation, causes),
            phases={
                name: score(errors, accepted, reasons, phase == i, causes)
                for i, name in enumerate(PHASES)
                if (phase == i).any()
            },
            semantic_latency_s=quantiles(np.asarray(latencies)),
            observer_wall_time_s=quantiles(elapsed),
            final_diagnostics=provider.diagnostics,
            recovery_s=provider.diagnostics.get("recovery_s"),
        )
        np.savez_compressed(
            output / "predictions.npz",
            errors=errors,
            accepted=accepted,
            reasons=reasons,
            causes=causes,
            confidence=confidence,
            phase=phase,
            time_s=observations["time_s"][:],
            frame=observations["frame"][:],
            components=np.array(COMPONENTS),
        )
        write_json(output / "report.json", report)
        write_json(output / "tracking.json", traces)
        write_json(
            output / "status.json",
            dict(
                state="finished",
                observations=n,
                offline_passed=report["manipulation"]["offline_passed"],
            ),
        )
    return report


def campaign_summary(reports, output, expected_episodes):
    doors = {}
    for asset in sorted({r["asset_id"] for r in reports}):
        selected = [r for r in reports if r["asset_id"] == asset]
        arrays = []
        for r in selected:
            with np.load(output / asset / r["condition"] / "predictions.npz") as p:
                arrays.append(
                    {k: p[k] for k in ("errors", "accepted", "reasons", "phase", "causes")}
                )
        joined = {k: np.concatenate([a[k] for a in arrays]) for k in arrays[0]}
        result = score(
            joined["errors"],
            joined["accepted"],
            joined["reasons"],
            np.isin(joined["phase"], MANIPULATION),
            joined["causes"],
        )
        doors[asset] = dict(
            split=selected[0]["split"],
            handedness=selected[0]["handedness"],
            **result,
            conditions={r["condition"]: r["manipulation"] for r in selected},
            phases={
                name: score(
                    joined["errors"],
                    joined["accepted"],
                    joined["reasons"],
                    joined["phase"] == i,
                    joined["causes"],
                )
                for i, name in enumerate(PHASES)
                if (joined["phase"] == i).any()
            },
            latency_by_condition={r["condition"]: r["semantic_latency_s"] for r in selected},
            recovery_by_condition={r["condition"]: r["recovery_s"] for r in selected},
        )
    complete = len(reports) == expected_episodes
    passed = complete and bool(doors) and all(d["offline_passed"] for d in doors.values())
    summary = dict(
        schema="b1.perception.prototype-evaluation.v1",
        episodes=len(reports),
        expected_episodes=expected_episodes,
        complete=complete,
        observations=sum(r["observations"] for r in reports),
        doors=doors,
        offline_passed=passed,
        passed_doors=sum(d["offline_passed"] for d in doors.values()),
        dynamic_status=(
            "pending_complete_campaign"
            if not complete
            else "eligible"
            if passed and expected_episodes == 50
            else "not_run_offline_gates_failed"
            if not passed
            else "pilot_only"
        ),
        training_started=False,
        collection_started=False,
        sealed_test_evaluated=False,
    )
    write_json(output / "summary.json", summary)
    lines = [
        "# Geometric B1 prototype evaluation",
        "",
        "Complete causal replay; fixed 1 cm / 5 degrees / 95% coverage / 95% precision.",
        f"Episodes: {len(reports)}/{expected_episodes}; observations: {summary['observations']}.",
        "Zero accepted estimates fail both rates. Missing errors are unobserved, never zero.",
        "",
        "| Door | Split | Hand | Coverage | Precision | Gates |",
        "|---|---|---|---:|---:|---|",
    ]
    for asset, d in doors.items():
        lines.append(
            f"| {asset} | {d['split']} | {d['handedness']} | {100 * d['valid_coverage']:.2f}% | "
            f"{100 * d['accepted_state_precision']:.2f}% | "
            f"{'pass' if d['offline_passed'] else 'fail'} |"
        )
    lines += [
        "",
        f"Passed doors: {summary['passed_doors']}/{len(doors)}.",
        f"Dynamic validation: {summary['dynamic_status']}.",
    ]
    for side in ("left", "right"):
        candidates = [(a, d) for a, d in doors.items() if d["handedness"] == side]
        if candidates:
            minimum = min(d["valid_coverage"] for _, d in candidates)
            lines += [
                "",
                f"Worst {side} coverage (including ties): "
                + ", ".join(a for a, d in candidates if d["valid_coverage"] == minimum)
                + ".",
            ]
            for key in (
                "panel_rotation_deg",
                "width_m",
                "height_m",
                "hinge_origin_m",
                "contact_position_m",
            ):
                finite = [
                    (d["errors_all_finite"][key]["p95"], a)
                    for a, d in candidates
                    if d["errors_all_finite"][key]["p95"] is not None
                ]
                if finite:
                    value, asset = max(finite)
                    lines.append(f"Worst {side} finite p95 {key}: {asset} = {value:.6f}.")
    for asset, d in doors.items():
        lines += [
            "",
            f"## {asset}",
            "",
            f"Expected manipulation observations: {d['observations']}; accepted: {d['accepted']}.",
            "Rejection causes (may overlap): "
            + ", ".join(f"{k}: {v}" for k, v in sorted(d["rejection_causes"].items()))
            + ".",
            "",
            "Finite errors include rejected partial estimates. Units are meters/degrees.",
            "",
            "| Component | Finite n | p50 | p95 | Maximum | Accepted n | Accepted p95 |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]

        def number(value):
            return "unobserved" if value is None else f"{value:.6f}"

        for key in COMPONENTS:
            all_, acc = d["errors_all_finite"][key], d["errors_accepted"][key]
            lines.append(
                f"| {key} | {all_['n']} | {number(all_['p50'])} | {number(all_['p95'])} | "
                f"{number(all_['maximum'])} | {acc['n']} | {number(acc['p95'])} |"
            )
        lines += [
            "",
            "| Condition / phase | Expected | Accepted | Coverage | Precision |",
            "|---|---:|---:|---:|---:|",
        ]
        for report in (r for r in reports if r["asset_id"] == asset):
            for phase, metrics in report["phases"].items():
                lines.append(
                    f"| {report['condition']} / {phase} | {metrics['observations']} | "
                    f"{metrics['accepted']} | {100 * metrics['valid_coverage']:.2f}% | "
                    f"{100 * metrics['accepted_state_precision']:.2f}% |"
                )
            lines += [
                "",
                f"{report['condition']} inference latency p50/p95/max (s): "
                + "/".join(
                    number(report["semantic_latency_s"][k]) for k in ("p50", "p95", "maximum")
                )
                + f"; recovery to supported tracking: {report['recovery_s']}; "
                f"reacquisitions: {report['final_diagnostics'].get('reacquisitions', 0)}.",
                f"All/rejected/accepted quantiles and per-phase causes: "
                f"[{report['condition']} JSON]({asset}/{report['condition']}/report.json).",
            ]
    (output / "report.md").write_text("\n".join(lines) + "\n")
    return summary
