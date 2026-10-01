"""Bounded scan diagnostics on existing recordings; no manipulation or release scoring."""

import hashlib
import json
import subprocess
import time
from dataclasses import asdict
from pathlib import Path

import h5py
import numpy as np
from PIL import Image, ImageDraw

from alexdoor_xas.perception.evaluation import quantiles, write_json
from alexdoor_xas.perception.geometry import project, surfaces
from alexdoor_xas.perception.scan import ScanMemory, footprint_support, surface_support
from alexdoor_xas.recording.b1 import OBS_KEYS

PILOTS = ("door-2738468b94d74c5f", "animated-door-1-88abf40")


def scan_description(state, config, distal_faces=None):
    objects = []
    for obj in state.objects:
        patches = []
        for patch in obj.patches:
            surface = next(s for s in obj.surfaces if s.surface_id == patch.surface_id)
            footprint = (
                footprint_support(surface, patch.pose, distal_faces, config)
                if distal_faces is not None
                else None
            )
            patches.append(
                dict(
                    patch_id=patch.patch_id,
                    surface_id=patch.surface_id,
                    position=patch.pose.origin,
                    rotation=patch.pose.rot,
                    support=asdict(patch.support),
                    ownership=patch.ownership,
                    footprint=asdict(footprint) if footprint else None,
                )
            )
        objects.append(
            dict(
                leaf_id=obj.leaf_id,
                ownership=obj.ownership,
                surfaces=[s.surface_id for s in obj.surfaces],
                membership_evidence={
                    s.surface_id: "registered_root_support"
                    if s is obj.root
                    else "observed_internal_connecting_seam"
                    for s in obj.surfaces
                },
                support=asdict(obj.support),
                observed_bounds=obj.root.bounds,
                observed_edges=obj.root.edges,
                edge_status={
                    key: sorted(
                        {o.edge_status.get(key, "unobserved") for o in obj.root.observations}
                    )
                    for key in ("width_0", "width_1", "height_0", "height_1")
                },
                hinges=[
                    dict(
                        candidate_id=h.candidate_id,
                        origin=h.origin,
                        direction=h.direction,
                        evidence=h.evidence,
                        reason=h.reason,
                        fit_residual_m=h.fit_residual_m,
                        condition=h.condition,
                        support=asdict(h.hypothesis.support) if h.hypothesis else None,
                    )
                    for h in obj.hinges
                ],
                patches=patches,
            )
        )
    retained = {s.surface_id: s for obj in state.objects for s in obj.surfaces}
    retained.update({s.surface_id: s for s in (*state.fixed_surfaces, *state.unresolved_surfaces)})
    return dict(
        generation=state.generation,
        reason=state.reason,
        objects=objects,
        fixed_surfaces=[s.surface_id for s in state.fixed_surfaces],
        unresolved_surfaces=[s.surface_id for s in state.unresolved_surfaces],
        surfaces=[
            dict(
                surface_id=s.surface_id,
                ownership=s.ownership,
                normal=s.normal,
                offset_m=s.offset,
                support=asdict(surface_support(s, state.generation, config)),
                observation_refs=sorted({(o.frame, o.mask_index) for o in s.observations}),
                observed_edges=s.edges,
            )
            for s in retained.values()
        ],
        ambiguity=dict(
            competing_leaf_candidates=len(state.objects),
            unresolved_parts=len(state.unresolved_surfaces),
            interpretation="Static continuity supports candidates, not motion identity; "
            "no area, color, mask score or asset label resolves alternatives.",
        ),
        unavailable_fields=["signed_articulation", "leaf_response", "stop_travel", "load_limits"],
        qualified=False,
        dynamic_state_implemented=False,
        bounds_model=(
            "recorded metric RGB-D/calibration; plane-tolerance depth error and "
            "fit conditioning; hardware calibration unqualified"
        ),
    )


def scan_overlay(path, sensor, memory, available_s):
    picture = Image.fromarray(sensor["rgb"])
    draw = ImageDraw.Draw(picture)
    h, w = sensor["rgb"].shape[:2]
    payload = {}
    palette = ((255, 180, 0), (30, 220, 255), (220, 90, 255), (120, 255, 100))
    members = {s.surface_id: i for i, o in enumerate(memory.state.objects) for s in o.surfaces}
    fixed = {s.surface_id for s in memory.state.fixed_surfaces}
    retained = {s.surface_id: s for s in (*memory.surfaces, *memory.state.unresolved_surfaces)}
    for index, surface in enumerate(retained.values()):
        color = (
            palette[members[surface.surface_id] % len(palette)]
            if surface.surface_id in members
            else (80, 160, 255)
            if surface.surface_id in fixed
            else (255, 80, 80)
        )
        cloud = np.r_[surface.points, surface.extent_points]
        pixels, z = project(cloud, sensor["intrinsics"], sensor["camera_world"])
        good = (z > 0) & np.isfinite(pixels).all(1)
        for u, v in pixels[good][:: max(1, int(good.sum()) // 1000)]:
            if 0 <= u < w and 0 <= v < h:
                draw.point((int(u), int(v)), fill=color)
        payload[f"surface_{index}"] = cloud
        payload[f"boundary_{index}"] = (
            np.concatenate([o.boundary_points for o in surface.observations])
            if surface.observations
            else np.empty((0, 3))
        )
        payload[f"surface_id_{index}"] = np.array(surface.surface_id)
    draw.text(
        (8, 8),
        f"Static candidates; available {available_s:.3f}s; {memory.state.reason}",
        fill="white",
        stroke_width=1,
        stroke_fill="black",
    )
    picture.save(path)
    np.savez_compressed(path.with_suffix(".npz"), **payload)


def save_cue(output, sensor, cue, row, provider):
    directory = output / "captures"
    directory.mkdir(exist_ok=True)
    name = f"{int(sensor['frame']):06d}"
    Image.fromarray(sensor["rgb"]).save(directory / f"{name}.png")
    np.savez_compressed(
        directory / f"{name}.npz",
        tokens=np.frombuffer(cue["tokens"], np.float32).reshape(cue["token_shape"]),
        masks=np.array([np.frombuffer(p, np.uint8) for p in cue["masks"]], np.uint8),
        scores=np.asarray(cue["scores"]),
        boxes=np.asarray(cue["boxes"]).reshape(-1, 4),
        pixel_mapping=np.array([cue["pixel_mapping"][k] for k in ("scale", "pad_x", "pad_y")]),
    )
    return dict(
        row=row,
        frame=int(sensor["frame"]),
        acquired_s=float(sensor["time_s"]),
        available_s=float(cue["available_s"]),
        image=f"captures/{name}.png",
        cue=f"captures/{name}.npz",
        latency_s=cue["latency_s"],
        reason=provider.scan_state.reason,
        objects=len(provider.scan_state.objects),
        surfaces=len(provider.static),
        fixed=len(provider.scan_state.fixed_surfaces),
        unresolved=len(provider.scan_state.unresolved_surfaces),
    )


def diagnose_scan(path, provider, output, distal_faces=None):
    """Every observed row through 25 s; pending results release without a fake RGB-D row."""
    path, output = Path(path), Path(output)
    output.mkdir(parents=True, exist_ok=False)
    provider.reset()
    records, last_frame, overlays = [], None, [0, *provider.config["inspection"]["sample_times_s"]]
    started = time.perf_counter()
    with h5py.File(path, "r") as h5:
        metadata = json.loads(h5["metadata"].attrs["episode"])
        if (
            metadata["asset_id"] not in PILOTS
            or metadata["split"] != "train"
            or metadata["condition"] not in ("nominal", "light")
        ):
            raise ValueError("Scan diagnostics are limited to the four authorized train pilots")
        observations = h5["observations"]
        times = observations["time_s"][:]
        stop = provider.config["inspection"]["sample_times_s"][-1]
        rows = np.flatnonzero(times <= stop + 1e-9)
        if not len(rows) or rows[-1] + 1 != len(rows) or not (np.diff(times[rows]) > 0).all():
            raise ValueError("Nonchronological or missing recorded scan")
        if abs(times[rows[0]]) > 1e-9 or times[rows[-1]] < stop - 1e-9:
            raise ValueError("Recording does not cover the complete fixed scan")
        row_lookup = {int(f): i for i, f in enumerate(observations["frame"][: len(rows)])}

        def record_result():
            nonlocal last_frame
            if provider.last_cue is None:
                return
            captured, cue = provider.last_cue
            frame = int(captured["frame"])
            if frame == last_frame:
                return
            last_frame = frame
            records.append(save_cue(output, captured, cue, row_lookup[frame], provider))
            if overlays and float(captured["time_s"]) >= overlays[0]:
                overlays.pop(0)
                scan_overlay(
                    output / f"scan-{frame:06d}.png",
                    captured,
                    provider.scan_memory,
                    cue["available_s"],
                )

        for row in rows:
            sensor = {key: observations[key][row] for key in OBS_KEYS}
            provider.update(sensor)
            record_result()
            if row % 600 == 0:
                status = dict(
                    state="running",
                    row=int(row),
                    scan_rows=len(rows),
                    semantic_results=len(records),
                    reason=provider.scan_state.reason,
                    elapsed_s=time.perf_counter() - started,
                )
                write_json(output / "status.json", status)
                print(
                    json.dumps(
                        dict(
                            asset_id=metadata["asset_id"], condition=metadata["condition"], **status
                        )
                    ),
                    flush=True,
                )
        # Replay inference has already computed the result; its measured completion still matters.
        if provider.engine.pending is not None:
            (_, captured, _), cue = provider.engine.pending
            ready = float(captured["time_s"]) + cue["latency_s"]
            provider.consume_results(ready)
            record_result()
        final_sensor = {key: observations[key][rows[-1]] for key in OBS_KEYS}
        available = max((r["available_s"] for r in records), default=float(times[rows[-1]]))
        scan_overlay(output / "scan-final.png", final_sensor, provider.scan_memory, available)
    report = dict(
        asset_id=metadata["asset_id"],
        condition=metadata["condition"],
        split="train",
        source=str(path),
        input_rows=len(rows),
        input_end_s=float(times[rows[-1]]),
        final_available_s=available,
        semantic_results=len(records),
        elapsed_s=time.perf_counter() - started,
        semantic_latency_s=quantiles(np.array([r["latency_s"] for r in records])),
        scan=scan_description(provider.scan_state, provider.config, distal_faces),
        complete=True,
        offline_passed=False,
        dynamic_passed=False,
        training_started=False,
        collection_started=False,
        sealed_test_evaluated=False,
        truth_inputs=False,
        annotations_read=False,
        manipulation_images_read=False,
    )
    write_json(output / "captures.json", records)
    write_json(output / "report.json", report)
    write_json(
        output / "status.json",
        dict(
            state="complete",
            input_rows=len(rows),
            semantic_results=len(records),
            reason=provider.scan_state.reason,
        ),
    )
    return report


def compare_video_scan(path, baseline, worker, config, distal_faces=None, *, output=None):
    """One retrospective association comparison on exactly the baseline captured RGB frames."""
    baseline = Path(baseline)
    output = baseline / "sam3-video" if output is None else Path(output)
    output.mkdir(parents=True, exist_ok=False)
    records = json.loads((baseline / "captures.json").read_text())
    prompts = None
    for index, record in enumerate(records):
        with np.load(baseline / record["cue"], allow_pickle=False) as data:
            if len(data["boxes"]):
                with Image.open(baseline / record["image"]) as image:
                    w, h = image.size
                boxes = []
                for x0, y0, x1, y1 in np.unique(data["boxes"], axis=0):
                    boxes.append(
                        [float(x0 / w), float(y0 / h), float((x1 - x0) / w), float((y1 - y0) / h)]
                    )
                prompts = dict(
                    frame_index=index, bounding_boxes=boxes, bounding_box_labels=[1] * len(boxes)
                )
                break
    if prompts is None:
        result = dict(state="not_run", reason="no_observed_grounding_box", qualified=False)
        write_json(output / "report.json", result)
        return result
    result = worker.infer_video([str(baseline / r["image"]) for r in records], prompts)
    write_json(output / "runtime.json", {k: v for k, v in result.items() if k != "frames"})
    memory = ScanMemory(config, 0)
    mask_counts = []
    with h5py.File(path, "r") as h5:
        observations = h5["observations"]
        for event in result.pop("frames"):
            record = records[event["frame_index"]]
            sensor = {key: observations[key][record["row"]] for key in OBS_KEYS}
            with np.load(baseline / record["cue"], allow_pickle=False) as data:
                mapping = dict(zip(("scale", "pad_x", "pad_y"), data["pixel_mapping"], strict=True))
                tokens = data["tokens"]
                cue = dict(
                    shape=sensor["rgb"].shape[:2],
                    tokens=tokens.tobytes(),
                    token_shape=tokens.shape,
                    pixel_mapping=mapping,
                    masks=event["masks"],
                    scores=event["scores"],
                    boxes=[],
                    latency_s=result["latency_s"],
                    available_s=max(r["acquired_s"] for r in records) + result["latency_s"],
                )
            memory.add(
                surfaces(cue, sensor, config),
                sensor,
                int(sensor["frame"]),
                available_s=cue["available_s"],
            )
            mask_counts.append(len(event["masks"]))
            np.savez_compressed(
                output / f"masks-{record['frame']:06d}.npz",
                masks=np.array([np.frombuffer(p, np.uint8) for p in event["masks"]], np.uint8),
                object_ids=event["object_ids"],
            )
        final = {key: observations[key][records[-1]["row"]] for key in OBS_KEYS}
        scan_overlay(
            output / "scan-final.png",
            final,
            memory,
            max(r["acquired_s"] for r in records) + result["latency_s"],
        )
    report = dict(result)
    report.update(
        state="complete",
        mode="retrospective_association_only",
        baseline=str(baseline),
        diagnosis="no_retained_video_masks" if not any(mask_counts) else memory.state.reason,
        frames=len(records),
        prompts=prompts,
        mask_counts=mask_counts,
        scan=scan_description(memory.state, config, distal_faces),
        qualified=False,
        provider_replaced=False,
        future_confirmation_not_backdated=True,
        training_started=False,
        collection_started=False,
        sealed_test_evaluated=False,
    )
    write_json(output / "report.json", report)
    return report


def run_scan_diagnostics(paths, recipe, output, models):
    from ihmc_alex_isaaclab._paths import REPOSITORY_ROOT as alex_root

    from alexdoor_xas.assets.purdue import derive_push_geometry
    from alexdoor_xas.perception.provider import CueEngine, GeometryProvider, ModelWorker

    expected = {(p, c) for p in PILOTS for c in ("nominal", "light")}
    if len(paths) != 4 or {(p.parent.parent.name, p.parent.name) for p in paths} != expected:
        raise ValueError("All four authorized pilot recordings are required")
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[3]
    urdf = (
        alex_root
        / "assets/robots/alex_purdue/urdf/baseline/alex_purdue_wsg32_umi_v1_full_convex.urdf"
    )
    geometry = derive_push_geometry(urdf)
    write_json(
        output / "protocol.json",
        dict(
            mode="scan_diagnostics",
            recipe=recipe.to_dict(),
            selected_episodes=[str(p) for p in paths],
            source_commit=subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=root, text=True
            ).strip(),
            provider_sources={
                p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                for p in Path(__file__).parent.glob("*.py")
            },
            input_end_s=recipe.config["inspection"]["sample_times_s"][-1],
            semantic_target_hz=1 / recipe.config["semantic_period_s"],
            footprint_source=str(urdf),
            footprint_vertices=[p.tolist() for p in geometry.distal_faces],
            video_comparison=(
                "one conditional forward configuration; same captured RGB "
                "frames and automatic GroundingDINO boxes; retrospective only"
            ),
            offline_passed=False,
            dynamic_passed=False,
            training_started=False,
            collection_started=False,
            sealed_test_evaluated=False,
            truth_boundary="metadata selects authorized recordings only; annotations never read",
        ),
    )
    reports, comparisons = [], []
    try:
        with (output / "models.log").open("w") as log:
            worker = ModelWorker(models, recipe.config, log=log)
            engine = CueEngine(worker, replay=True)
            try:
                write_json(output / "runtime.json", worker.runtime)
                # One CUDA warmup on an authorized scan observation; discard outputs.
                with h5py.File(paths[0], "r") as h5:
                    worker.infer(h5["observations/rgb"][0])
                provider = GeometryProvider(recipe, engine)
                for path in paths:
                    destination = output / path.parent.parent.name / path.parent.name
                    reports.append(
                        diagnose_scan(path, provider, destination, geometry.distal_faces)
                    )
                    print(
                        json.dumps(
                            dict(
                                completed_scans=len(reports),
                                scans=4,
                                reason=reports[-1]["scan"]["reason"],
                            )
                        ),
                        flush=True,
                    )
            finally:
                engine.close()
                worker.close()
        conditional = any(r["scan"]["reason"] != "supported_static_candidate" for r in reports)
        if conditional:
            with (output / "video-models.log").open("w") as log:
                worker = ModelWorker(
                    models, dict(recipe.config, worker_mode="video-diagnostic"), log=log
                )
                try:
                    write_json(output / "video-runtime.json", worker.runtime)
                    for path in paths:
                        destination = output / path.parent.parent.name / path.parent.name
                        comparisons.append(
                            compare_video_scan(
                                path, destination, worker, recipe.config, geometry.distal_faces
                            )
                        )
                        print(
                            json.dumps(
                                dict(
                                    completed_video_comparisons=len(comparisons),
                                    comparisons=4,
                                    state=comparisons[-1]["state"],
                                )
                            ),
                            flush=True,
                        )
                finally:
                    worker.close()
        summary = dict(
            complete=True,
            completed_scans=len(reports),
            expected_scans=4,
            object_reasons=[r["scan"]["reason"] for r in reports],
            video_triggered=conditional,
            completed_video_comparisons=len(comparisons),
            video_reasons=[
                r["scan"]["reason"] if "scan" in r else r["reason"] for r in comparisons
            ],
            offline_passed=False,
            dynamic_passed=False,
            training_started=False,
            collection_started=False,
            sealed_test_evaluated=False,
            provider_replaced=False,
        )
        write_json(output / "summary.json", summary)
        print(json.dumps(summary), flush=True)
        return 0
    except (Exception, KeyboardInterrupt) as error:
        write_json(
            output / "failure.json",
            dict(
                complete=False,
                completed_scans=len(reports),
                completed_video_comparisons=len(comparisons),
                error=f"{type(error).__name__}: {error}",
                offline_passed=False,
                dynamic_passed=False,
            ),
        )
        raise
