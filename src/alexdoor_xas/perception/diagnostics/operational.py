"""Complete provider accounting under continuous ingress; truth stays in scoring."""

import json
import subprocess
import time
from pathlib import Path

import h5py
import numpy as np

from alexdoor_xas.perception.diagnostics.offline import finite_pose
from alexdoor_xas.perception.diagnostics.smoke import sensor_at
from alexdoor_xas.perception.diagnostics.targets import evaluate_targets
from alexdoor_xas.perception.evaluation import json_safe, quantiles, write_json
from alexdoor_xas.perception.ipc import unpack_array
from alexdoor_xas.perception.point2pose.tracking import measured_registration
from alexdoor_xas.perception.session import session_provider


class OperationalRecorder:
    """Retain every input and publication, including unsupported and replaced captures."""

    def __init__(self, output):
        self.output = Path(output)
        self.inputs, self.events = [], []
        self.input_stream = (self.output / "inputs.jsonl").open("w", buffering=1)
        self.event_stream = (self.output / "events.jsonl").open("w", buffering=1)
        self.mask_stream = (self.output / "masks-packed.bin").open("wb")

    def input(self, sensor, received_s, update_s, **extra):
        row = dict(
            frame=int(sensor["frame"]),
            time_s=float(sensor["time_s"]),
            received_s=received_s,
            provider_update_s=update_s,
            **extra,
        )
        self.inputs.append(row)
        self.input_stream.write(json.dumps(json_safe(row)) + "\n")

    def publication(self, provider):
        if provider.last_publication is None:
            return
        captured, result = provider.last_publication
        if self.events and self.events[-1]["frame"] == result["frame"]:
            return
        objects = []
        for candidate, raw in zip(provider.tracking.candidates, result["objects"], strict=True):
            obj = {
                k: unpack_array(v) if isinstance(v, dict) and "data" in v else v
                for k, v in raw.items()
            }
            source, p, a = measured_registration(obj, provider.config["plane_tolerance_m"])
            reasons = []
            if obj["lost"]:
                reasons.append("native_tracking_lost")
            if result["semantic_absent"]:
                reasons.append("selected_identity_absent")
            if p is None or a is None:
                reasons.append(
                    "insufficient_measured_pairs" if len(source) < 5 else "degenerate_registration"
                )
            valid = finite_pose(obj["camera_from_map"])
            if not valid:
                reasons.append("invalid_native_pose")
            objects.append(
                dict(
                    object_id=obj["object_id"],
                    camera_from_map=obj["camera_from_map"],
                    native_pose_valid=valid,
                    native_lost=obj["lost"],
                    native_tracking=valid and not obj["lost"],
                    integration_accepted=candidate.state == "tracked" and not reasons,
                    integration_rejections=reasons,
                    measured_pairs=len(source),
                    registration_position_m=p,
                    registration_rotation_rad=a,
                )
            )
        packed = b"".join(result["cue"]["masks"])
        mask_store = dict(
            offset=self.mask_stream.tell(),
            size=len(packed),
            shape=result["cue"]["shape"],
            ids=result["cue"]["ids"],
        )
        self.mask_stream.write(packed)
        if (
            result["capture_s"] >= provider.config["inspection"]["sample_times_s"][-1]
            and not (self.output / "selection-view.npz").exists()
        ):
            masks = np.asarray(
                [
                    np.unpackbits(np.frombuffer(m, np.uint8), count=np.prod(result["cue"]["shape"]))
                    .reshape(result["cue"]["shape"])
                    .astype(bool)
                    for m in result["cue"]["masks"]
                ]
            )
            np.savez_compressed(self.output / "selection-view.npz", **captured, masks=masks)
            from PIL import Image

            Image.fromarray(captured["rgb"]).save(self.output / "selection-view.png")
        seed = bool(result["initial_candidates"])
        row = dict(
            frame=result["frame"],
            time_s=result["capture_s"],
            objects=objects,
            available_s=provider.diagnostics["available_time_s"],
            observation_age_s=result["publication_latency_s"],
            is_seed=seed,
            identity=result["identity"],
            initialization=result["initialization"],
            geometric_choices=result["geometric_choices"],
            geometry_candidates=result.get("geometry_candidates", []),
            initialization_rejections=result.get("initialization_rejections", []),
            mask_store=mask_store,
            sam3=result["sam3"],
            stages=result["stages"],
            service_s=result["latency_s"],
            publication_s=result["provider_publication_s"],
            queue_wait_s=result.get("queue_wait_s", 0),
            ingress_delay_s=result.get("ingress_delay_s", 0),
            executor_wait_s=result.get("executor_wait_s", 0),
            poll_wait_s=result.get("publication_poll_wait_s", 0),
            dispatch_s=result.get("dispatch_s"),
            compute_completed_s=result.get("compute_completed_s"),
            snapshot_copy_s=result.get("snapshot_copy_s", 0),
            queue=provider.engine.queue_stats.copy(),
            resources=result.get("resources"),
            binding=result.get("binding"),
        )
        self.events.append(json_safe(row))
        self.event_stream.write(json.dumps(json_safe(row)) + "\n")

    def summary(self, provider):
        seeds = [e for e in self.events if e["is_seed"]]
        seed_s = seeds[0]["time_s"] if seeds else None
        groups = dict(
            inspection=[e for e in self.events if not e["objects"]],
            initialization=seeds,
            warmup=[
                e
                for e in self.events
                if e["objects"] and not e["is_seed"] and e["time_s"] < seed_s + 5
            ],
            steady=[
                e
                for e in self.events
                if e["objects"] and not e["is_seed"] and e["time_s"] >= seed_s + 5
            ],
        )
        fields = (
            "service_s",
            "publication_s",
            "queue_wait_s",
            "ingress_delay_s",
            "executor_wait_s",
            "poll_wait_s",
            "observation_age_s",
            "snapshot_copy_s",
        )
        phases = {}
        for name, events in groups.items():
            phases[name] = dict(
                publications=len(events),
                timing_s={key: quantiles(np.asarray([e[key] for e in events])) for key in fields},
                stages_s={
                    key: quantiles(np.asarray([e["stages"][key] for e in events]))
                    for key in (
                        "sam3_s",
                        "geometry_and_seed_s",
                        "point2pose_s",
                        "identity_binding_s",
                    )
                },
                under_150ms=sum(e["observation_age_s"] <= 0.15 for e in events),
                publication_hz=(len(events) - 1)
                / (events[-1]["available_s"] - events[0]["available_s"])
                if len(events) > 1
                else None,
                publication_intervals_s=quantiles(np.diff([e["available_s"] for e in events])),
                measured_support_refusals=sum(
                    bool(o["integration_rejections"]) for e in events for o in e["objects"]
                ),
            )
        processed = {e["frame"] for e in self.events}
        skipped = [i["frame"] for i in self.inputs if i["frame"] not in processed]
        write_json(self.output / "skipped-frame-ids.json", skipped)
        samples = []
        for path in self.output.glob("session-*/memory-samples.jsonl"):
            samples.extend(json.loads(x) for x in path.read_text().splitlines())
        return dict(
            input_frames=len(self.inputs),
            processed_frames=len(processed),
            skipped_frames=len(skipped),
            queue=provider.engine.queue_stats.copy(),
            phases=phases,
            provider_update_s=quantiles(np.asarray([i["provider_update_s"] for i in self.inputs])),
            simultaneous_gpu_peak_bytes=max((s.get("gpu_bytes", 0) for s in samples), default=None),
            simultaneous_rss_peak_bytes=max((s.get("rss_bytes", 0) for s in samples), default=None),
            sample_errors=sum("error" in s for s in samples),
            resource_period_s=0.25,
            resource_scope=(
                "coordinator including renderer, SAM3 and P2P; simultaneous post-ready samples"
            ),
            startup_spikes_qualified=False,
            seed_capture_s=seed_s,
            qualified=False,
            loaded_contact_admitted=False,
        )

    def close(self):
        self.input_stream.close()
        self.event_stream.close()
        self.mask_stream.close()


def recorded_operational(path, recipe, output, models, *, selection=None, replay=False):
    """One complete original recording; real-time ingress or measured-service replay."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    recorder = OperationalRecorder(output)
    provider = None
    epoch = None
    with h5py.File(path, "r") as h5:
        calibration = json.loads(h5["metadata"].attrs["calibration"])
        times = h5["observations/time_s"][:]
        frame_ids = h5["observations/frame"][:]

        def clock():
            return float(times[0]) + time.perf_counter() - epoch

        write_json(
            output / "protocol.json",
            dict(
                mode="measured-service replay"
                if replay
                else "continuous recorded wall-clock ingress",
                source=str(path),
                source_commit=subprocess.check_output(
                    ["git", "rev-parse", "HEAD"], text=True
                ).strip(),
                recipe=recipe.to_dict(),
                selection=selection,
                source_times_unchanged=True,
                clock="original capture origin + monotonic wall elapsed; boot precedes acquisition",
                annotations="post-run evaluator only",
                registration_diagnostics=True,
            ),
        )
        try:
            boot = time.perf_counter()
            provider = session_provider(
                recipe,
                models,
                calibration,
                output,
                replay=replay,
                selection=selection,
                registration_diagnostics=True,
                clock=None if replay else clock,
            )
            startup_s = time.perf_counter() - boot
            epoch = time.perf_counter()
            for row, t in enumerate(times):
                if not replay:
                    time.sleep(max(0, float(t) - clock()))
                sensor = sensor_at(h5, row)
                received = float(t) if replay else clock()
                started = time.perf_counter()
                provider.update(sensor, now=received)
                recorder.input(sensor, received, time.perf_counter() - started, row=row)
                recorder.publication(provider)
                if provider.engine.failed:
                    raise RuntimeError(provider.diagnostics.get("error", "session_failed"))
            drain_started = time.perf_counter()
            while provider.engine.pending is not None:
                if time.perf_counter() - drain_started > 30:
                    raise TimeoutError("operational_drain_timeout")
                now = provider.engine.pending[1]["available_s"] if replay else clock()
                provider.consume_results(now)
                recorder.publication(provider)
                if not replay:
                    time.sleep(0.005)
            summary = recorder.summary(provider)
            summary.update(
                complete=True, startup_s=startup_s, wall_elapsed_s=time.perf_counter() - epoch
            )
            lookup = {int(f): i for i, f in enumerate(frame_ids)}
            events = [dict(e, row=lookup[e["frame"]]) for e in recorder.events]
            seed_row = next((e["row"] for e in events if e["is_seed"]), None)
            # The existing physical targets are evaluator inputs, never worker inputs.
            frozen_path = (
                Path(__file__).resolve().parents[4]
                / "outputs/evidence/b1/perception/p2p-performance-01/full-target-evaluation.json"
            )
            if Path(path).parent.parent.name == "animated-door-1-88abf40" and frozen_path.exists():
                frozen = json.loads(frozen_path.read_text())[Path(path).parent.name]["chunkall"]
                targets = [d["definition"]["seed_target"] for d in frozen["targets"]]
                opening = [e for e in events if e["time_s"] >= 31 - 1e-9]
                by_frame = {e["frame"]: e for e in events}
                all_opening = [
                    by_frame.get(
                        int(frame_ids[i]),
                        dict(row=i, frame=int(frame_ids[i]), time_s=float(t), objects=[]),
                    )
                    for i, t in enumerate(times)
                    if t >= 31 - 1e-9
                ]
                summary["quality"] = dict(
                    processed_only=evaluate_targets(
                        h5, opening, targets, reference_row=1860, seed_row=seed_row
                    ),
                    all_scheduled_with_skips=evaluate_targets(
                        h5, all_opening, targets, reference_row=1860, seed_row=seed_row
                    ),
                    scope=(
                        "same three frozen physical targets; capture-time quality of "
                        "actual processed outputs; skipped unavailable"
                    ),
                )
            write_json(output / "report.json", summary)
            old = provider.engine.worker
            old_pids = [w.process.pid for w in (old.sam, old.native)]
            provider.reset(preload=False)
            write_json(
                output / "reset.json",
                dict(
                    old_pids=old_pids,
                    old_processes_exited=all(
                        w.process.poll() is not None for w in (old.sam, old.native)
                    ),
                    empty_candidates=not provider.tracking.candidates,
                    empty_selection=provider.tracking.selection is None,
                    empty_worker=provider.engine.worker is None,
                    empty_publication=provider.last_publication is None,
                    empty_queue=provider.engine.pending is provider.engine.latest is None,
                    generation=provider.generation,
                ),
            )
            return summary
        except BaseException as error:
            write_json(
                output / "failure.json",
                dict(
                    error=f"{type(error).__name__}: {error}",
                    inputs=len(recorder.inputs),
                    publications=len(recorder.events),
                ),
            )
            raise
        finally:
            recorder.close()
            if provider is not None:
                provider.close()
