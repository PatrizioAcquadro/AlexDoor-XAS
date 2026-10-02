#!/usr/bin/env python
"""Bounded 6.0C material replay/live diagnosis; never qualify or collect learning data."""

import argparse
import json
import os
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "4")
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
PILOTS = ("door-2738468b94d74c5f", "animated-door-1-88abf40")


def configure(provider, covers, asset, condition):
    import numpy as np

    from alexdoor_xas.perception.contracts import LocalContactSelection

    source = REPO / "outputs/b1/perception/operational-contact-readiness-01"
    rows = json.loads((source / f"{asset}-{condition}-candidates.json").read_text())
    reference_tangent = (-0.1, 1.5) if asset == PILOTS[1] else (0.0, 1.1)
    selected_tangent = (0.0, 1.1)
    # Diagnostic requests are explicit; no asset/handedness reaches the provider.
    tangents = [reference_tangent, selected_tangent, (-0.1, 1.3)]
    tangents = list(dict.fromkeys(tangents))
    tracker = provider.configure_material(covers)
    mapping = {}
    for tangent in tangents:
        candidate = next((r for r in rows if np.allclose(r["position"][1:], tangent)), None)
        if candidate is None:
            continue
        patch_id = f"candidate-{len(mapping)}"
        tracker.request(patch_id, candidate["position"], reference=tangent == reference_tangent)
        mapping[patch_id] = candidate
    contact_id = next(
        k for k, r in mapping.items() if np.allclose(r["position"][1:], selected_tangent)
    )
    return tracker, mapping, contact_id, LocalContactSelection


def compare_candidates(mapping, calibration, q, chain):
    import numpy as np
    import torch

    base = np.asarray(calibration["root_world"])
    reports = []
    for patch_id, candidate in mapping.items():
        position, rotation = np.array(candidate["position"]), np.array(candidate["rotation"])
        seed = chain.array(q[:7]).reshape(1, 7)
        endpoints = []
        for distance in (0.03, 0):
            world = position - distance * rotation[:, 0]
            p, r = (world - base[:3, 3]) @ base[:3, :3], base[:3, :3].T @ rotation
            seed, pe, re = chain.solve(chain.array(p)[None], chain.array(r)[None], seed)
            _, jac = chain.forward(seed)
            endpoints.append(
                dict(
                    precontact_distance_m=distance,
                    position_error_m=pe.item(),
                    orientation_error_rad=re.item(),
                    joint_limit_margin_rad=float(
                        torch.minimum(seed - chain.limits[:, 0], chain.limits[:, 1] - seed).min()
                    ),
                    jacobian_min_singular=float(torch.linalg.svdvals(jac)[0, -1]),
                    joints=seed[0].tolist(),
                )
            )
        reports.append(
            dict(
                patch_id=patch_id,
                tangent_coordinates=position[1:].tolist(),
                endpoint_reachable=all(
                    e["position_error_m"] <= 0.01 and e["orientation_error_rad"] <= np.deg2rad(5)
                    for e in endpoints
                ),
                endpoints=endpoints,
                observed_clearance_m=candidate["score_clearance_m"],
                path_collision_checked=False,
                pose_quality_validated=False,
                definitive_contact=False,
            )
        )
    return reports


def trace_record(estimate, provider, sensor, row):
    from alexdoor_xas.perception.contracts import validate_local_contact

    now = float(sensor["time_s"])
    state = estimate.local
    supported, reason = False, "unavailable_local_state"
    if state is not None:
        try:
            validate_local_contact(state, now, generation=provider.generation)
            supported, reason = True, "supported_local_geometry_only"
        except (ValueError, TypeError, AttributeError) as error:
            reason = str(error)
    return dict(
        row=row,
        frame=int(sensor["frame"]),
        time_s=now,
        local=asdict(state) if state is not None else None,
        selected_geometry_supported=supported,
        selected_reason=reason,
        processing_s=provider.diagnostics.get("material_processing_s"),
        operational_state_produced=estimate.operational is not None,
        tracks={
            k: dict(
                **t.tracker.diagnostics, reason=t.reason, losses=t.losses, recoveries=t.recoveries
            )
            for k, t in provider.material.tracks.items()
        },
    )


def run_replay(args):
    import h5py
    import numpy as np
    import torch
    from ihmc_alex_isaaclab._paths import REPOSITORY_ROOT as alex_root

    from alexdoor_xas.assets.purdue import derive_push_geometry
    from alexdoor_xas.kinematics.purdue_chain import PurdueChain
    from alexdoor_xas.perception.evaluation import json_safe, write_json
    from alexdoor_xas.perception.provider import (
        CueEngine,
        GeometryProvider,
        ModelWorker,
        load_recipe,
    )
    from alexdoor_xas.recording.b1 import OBS_KEYS

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required; no CPU fallback")
    urdf = (
        alex_root
        / "assets/robots/alex_purdue/urdf/baseline/alex_purdue_wsg32_umi_v1_full_convex.urdf"
    )
    covers, chain = derive_push_geometry(urdf).contact_covers(), PurdueChain(urdf)
    recipe = load_recipe(REPO / "configs/perception_geometry.json", REPO)
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(
        args.output / "protocol.json",
        dict(
            source_commit=subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=REPO, text=True
            ).strip(),
            recipe=recipe.to_dict(),
            mode="bounded_material_replay",
            visual_reference_separate_from_contact=True,
            explicit_diagnostic_contact_tangent=[0.0, 1.1],
            contact_is_definitive=False,
            saved_scan_end_s=25,
            availability="saved image events plus measured extraction/tracking completion",
            annotations_read=False,
            offline_passed=False,
            dynamic_passed=False,
        ),
    )
    reports = []
    with (args.output / "models.log").open("w") as log:
        worker = ModelWorker(REPO / "models/perception", recipe.config, log)
        try:
            write_json(args.output / "runtime.json", worker.runtime)
            for asset in PILOTS:
                for condition in ("nominal", "light"):
                    if args.asset and (args.asset != asset or args.condition != condition):
                        continue
                    folder = args.output / asset / condition
                    folder.mkdir(parents=True)
                    source = (
                        REPO
                        / f"datasets/b1/perception/engineering-v2/{asset}/{condition}/episode.hdf5"
                    )
                    captures = (
                        REPO / f"outputs/b1/perception/operational-scan-02/{asset}/{condition}"
                    )
                    records = {
                        r["frame"]: r for r in json.loads((captures / "captures.json").read_text())
                    }

                    class SavedScanEngine(CueEngine):
                        def submit(self, sensor, view, records=records, captures=captures):
                            if self.pending is not None:
                                return False
                            if float(sensor["time_s"]) > 25:
                                return super().submit(sensor, view)
                            record = records.get(int(sensor["frame"]))
                            if record is None:
                                return False
                            with np.load(captures / record["cue"], allow_pickle=False) as data:
                                scale, px, py = data["pixel_mapping"]
                                cue = dict(
                                    shape=sensor["rgb"].shape[:2],
                                    masks=[m.tobytes() for m in data["masks"]],
                                    tokens=data["tokens"].tobytes(),
                                    token_shape=data["tokens"].shape,
                                    scores=data["scores"].tolist(),
                                    boxes=data["boxes"].tolist(),
                                    pixel_mapping=dict(scale=scale, pad_x=px, pad_y=py),
                                    latency_s=record["available_s"] - record["acquired_s"],
                                    available_s=record["available_s"],
                                )
                            self.pending = (
                                (
                                    self.generation,
                                    {k: np.array(v, copy=True) for k, v in sensor.items()},
                                    view,
                                ),
                                cue,
                            )
                            return True

                    engine = SavedScanEngine(worker, replay=True)
                    provider = GeometryProvider(recipe, engine)
                    tracker, mapping, contact_id, selection_type = configure(
                        provider, covers, asset, condition
                    )
                    counts, axes, supported = {}, 0, 0
                    with h5py.File(source, "r") as h5, (folder / "trace.jsonl").open("w") as trace:
                        metadata = json.loads(h5["metadata"].attrs["episode"])
                        if metadata["split"] != "train":
                            raise ValueError("Only the two train pilots are authorized")
                        calibration = json.loads(h5["metadata"].attrs["calibration"])
                        observations = h5["observations"]
                        write_json(
                            folder / "candidates.json",
                            compare_candidates(
                                mapping, calibration, observations["joint_position"][0], chain
                            ),
                        )
                        n = len(observations["time_s"])
                        for row in range(n):
                            sensor = {key: observations[key][row] for key in OBS_KEYS}
                            estimate = provider.update(sensor)
                            if (
                                tracker.selection is None
                                and tracker.tracks[contact_id].geometry is not None
                                and tracker.tracks[contact_id].geometry.available_s
                                <= float(sensor["time_s"])
                            ):
                                tracker.select(
                                    selection_type(
                                        "diagnostic-selection-1",
                                        contact_id,
                                        float(sensor["time_s"]),
                                        "diagnostic",
                                    ),
                                    float(sensor["time_s"]),
                                )
                            record = trace_record(estimate, provider, sensor, row)
                            trace.write(json.dumps(json_safe(record), allow_nan=False) + "\n")
                            counts[record["selected_reason"]] = (
                                counts.get(record["selected_reason"], 0) + 1
                            )
                            supported += record["selected_geometry_supported"]
                            axes += bool(estimate.local and estimate.local.hypotheses)
                            if row % 600 == 0:
                                write_json(
                                    folder / "status.json", dict(row=row, rows=n, state="running")
                                )
                                print(
                                    asset, condition, row, n, record["selected_reason"], flush=True
                                )
                        report = dict(
                            asset=asset,
                            condition=condition,
                            rows=n,
                            input_end_s=float(sensor["time_s"]),
                            selected_supported_ticks=supported,
                            axis_ticks=axes,
                            reasons=counts,
                            tracks={
                                k: dict(
                                    initialized=t.geometry is not None,
                                    losses=t.losses,
                                    recoveries=t.recoveries,
                                    last_support_s=(t.dynamic.supported_s if t.dynamic else None),
                                )
                                for k, t in tracker.tracks.items()
                            },
                            offline_passed=False,
                            dynamic_passed=False,
                            training_started=False,
                            collection_started=False,
                            sealed_test_evaluated=False,
                            annotations_read=False,
                            mode="bounded_material_replay",
                        )
                        write_json(folder / "report.json", report)
                        reports.append(report)
                    engine.close()
            write_json(
                args.output / "summary.json",
                dict(
                    reports=reports,
                    complete=len(reports) == 4,
                    offline_passed=False,
                    dynamic_passed=False,
                ),
            )
        except BaseException as error:
            write_json(
                args.output / "failure.json",
                dict(
                    error=f"{type(error).__name__}: {error}",
                    completed_episodes=len(reports),
                    complete=False,
                    offline_passed=False,
                    dynamic_passed=False,
                ),
            )
            raise
        finally:
            worker.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("replay", "live"))
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--asset", choices=PILOTS)
    parser.add_argument("--condition", choices=("nominal", "light"))
    args = parser.parse_args()
    if bool(args.asset) != bool(args.condition):
        parser.error("An individual case requires both --asset and --condition")
    if args.mode == "replay":
        run_replay(args)
    else:
        from alexdoor_xas.perception.material_live import run_live

        run_live(args, REPO)


if __name__ == "__main__":
    main()
