"""One causal SAM3 coordinator for inspection and externally masked Point2Pose.

The coordinator owns temporal state. Scheduling lives in the bounded engine;
serial evaluation calls the same coordinator directly, without dropping captures.
"""

import json
import threading
import time
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import numpy as np

from alexdoor_xas.perception.geometry import surfaces
from alexdoor_xas.perception.point2pose.tracking import PanelTracking
from alexdoor_xas.perception.scan import (
    ObservedSpace,
    ScanMemory,
    registered_support,
    surface_support,
)


def selected_recipe(path=None):
    path = (
        Path(path)
        if path
        else Path(__file__).resolve().parents[3] / "configs/point2pose_selected.json"
    )
    return json.loads(path.read_text())


class SerialTrackingEngine:
    """Native IPC without operational availability/queue scheduling."""

    def __init__(self, worker=None):
        self.worker, self.result = worker, None

    def reset(self):
        self.result = None

    def submit(self, sensor, masks=None, *, start_s=None):
        self.result = self.worker.infer(sensor, masks)
        if self.result["frame"] != int(sensor["frame"]) or self.result["capture_s"] != float(
            sensor["time_s"]
        ):
            raise ValueError("point2pose_result_capture_mismatch")
        return True


class PerceptionSession:
    """Single selected object; no ID substitution or periodic text prompting.

    Inspect only the prescribed stationary window. Freeze that map at its end,
    even if initialization must wait for a later supported acquisition. P2P's
    reference/zone is measured on its actual seed, with independent support times.
    """

    def __init__(
        self,
        models,
        recipe,
        calibration,
        output,
        *,
        tracking=True,
        selected=None,
        initialize_after_s=None,
        registration_diagnostics=False,
        sam_worker=None,
        p2p_worker=None,
        historical_selection=False,
        generation=0,
    ):
        from alexdoor_xas.perception.point2pose.runtime import Point2PoseWorker
        from alexdoor_xas.perception.sam3.runtime import Sam3Worker

        self.config = recipe.config
        self.selected = selected or self.config.get("point2pose_selected") or selected_recipe()
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        self.log = (self.output / "sam3.log").open("a")
        self.sam = self.native = self.resources = None
        try:
            self.sam = sam_worker or Sam3Worker(models, self.selected["sam3_frontend"], self.log)
            if tracking:
                self.native = p2p_worker or Point2PoseWorker(
                    Path(models) / "point2pose",
                    calibration["depth_interval_m"],
                    self.config["plane_tolerance_m"],
                    1,
                    self.output / "native",
                    external_masks=True,
                    registration_diagnostics=registration_diagnostics,
                    **self.selected["point2pose_controls"],
                )
                self.native.first_request = False
        except BaseException:
            self.close()
            raise
        workers = [w for w in (self.sam, self.native) if w is not None]
        if all(hasattr(w, "process") and hasattr(w.process, "pid") for w in workers):
            from alexdoor_xas.perception.resources import SessionResources

            self.resources = SessionResources(workers, self.output / "memory-samples.jsonl")
        self.engine = SerialTrackingEngine(self.native)
        self.tracker = PanelTracking(self.engine, self.config)
        self.tracker.reset(generation)
        self.scan_memory = ScanMemory(self.config, generation)
        self.inspection_end = self.config["inspection"]["sample_times_s"][-1]
        self.initialize_after = (
            self.inspection_end if initialize_after_s is None else initialize_after_s
        )
        self.next_geometry = self.config["inspection"]["sample_times_s"][0]
        self.identity = None
        self.historical_selection = historical_selection
        self.lock = threading.Lock()
        self.requested_identity = None
        self.requested_surface = None
        self.requested_pixel = None
        self.selection_after_s = 0.0
        self.last_capture = None
        self.last_present = None
        self.runtime = dict(
            sam3=self.sam.runtime,
            point2pose=None if self.native is None else self.native.runtime,
            initialize_after_s=self.initialize_after,
            static_fusion_ends_s=self.inspection_end,
            diagnostic_only=False,
            qualified=False,
        )
        self.boot_latency_s = self.sam.runtime.get("startup_s", 0) + (
            self.native.boot_latency_s if self.native else 0
        )

    def select(
        self,
        identity,
        geometric_object=None,
        *,
        reference_pixel=None,
        after_s=0.0,
        criterion="explicit API selection",
    ):
        """Request a semantic ID; latch on a processed acquisition before initialization."""
        with self.lock:
            identity = int(identity)
            if any(
                i is not None and i != identity for i in (self.identity, self.requested_identity)
            ):
                raise ValueError("selection_change_requires_session_reset")
            if geometric_object is not None and reference_pixel is not None:
                raise ValueError("select_one_geometric_reference")
            pixel = None if reference_pixel is None else tuple(reference_pixel)
            if pixel is not None and (
                len(pixel) != 2 or any(not isinstance(v, int) or v < 0 for v in pixel)
            ):
                raise ValueError("invalid_reference_pixel")
            if not np.isfinite(after_s) or after_s < 0:
                raise ValueError("invalid_selection_time")
            if self.tracker.candidates and (
                geometric_object != self.requested_surface or pixel != self.requested_pixel
            ):
                raise ValueError("selection_change_requires_session_reset")
            self.requested_identity = identity
            self.requested_surface = geometric_object
            self.requested_pixel = pixel
            self.selection_after_s = float(after_s)
            self.runtime["selection_request"] = dict(
                sam3_id=identity,
                geometric_object=geometric_object,
                reference_pixel=pixel,
                after_s=float(after_s),
                ownership_qualified=False,
                criterion=criterion,
            )

    def infer(self, sensor, masks=None):
        if masks is not None:
            raise ValueError("session_owns_synchronized_masks")
        started = time.perf_counter()
        frame, t = int(sensor["frame"]), float(sensor["time_s"])
        if self.last_capture is not None and (
            frame <= self.last_capture[0] or t <= self.last_capture[1]
        ):
            raise ValueError("nonmonotonic_session_capture")
        self.last_capture = frame, t
        if self.resources:
            self.resources.capture_s = t
        cue = self.sam.infer(sensor["rgb"], frame, t)
        meta = cue["sam3"]
        if meta["frame"] != frame or meta["time_s"] != t:
            raise ValueError("unsynchronized_sam3_mask")
        ids = cue["ids"]
        with self.lock:
            if self.identity is None:
                if self.requested_identity is not None:
                    self.identity = self.requested_identity
                elif len(ids) == 1:
                    self.identity = ids[0]
        present = self.identity is not None and self.identity in ids
        state = (
            ("no_candidate" if not ids else "selection_required")
            if self.identity is None
            else ("selected" if present else "selected_identity_absent")
        )
        returned = present and self.last_present is False
        self.last_present = present if self.identity is not None else None
        geometric_choices = []
        geometry_candidates = []
        reference_state = "not_inspected"
        native_result = None
        initial_candidates = None
        scan_snapshot = None
        geometry_s = 0.0
        if not self.tracker.candidates:
            if t >= self.next_geometry or t >= self.initialize_after:
                geometry_started = time.perf_counter()
                candidates = surfaces(cue, sensor, self.config)
                for index, surface in enumerate(candidates):
                    surface.surface_id = (
                        f"sam3-{surface.observations[0].semantic_id}:surface-{index}"
                    )
                geometry_candidates = [
                    dict(
                        surface_id=s.surface_id,
                        sam3_id=s.observations[0].semantic_id,
                        normal=s.normal.tolist(),
                        extent_m=np.ptp(s.points @ s.basis, axis=0).tolist(),
                        points=len(s.points),
                    )
                    for s in candidates
                ]
                if t <= self.inspection_end:
                    self.scan_memory.add(candidates, sensor, frame)
                    scan_snapshot = deepcopy(self.scan_memory)
                self.next_geometry = t + self.config["semantic_period_s"]
                if self.native is not None and present:
                    selected_surfaces = [
                        s for s in candidates if s.observations[0].semantic_id == self.identity
                    ]
                    with self.lock:
                        requested_surface = self.requested_surface
                        requested_pixel = self.requested_pixel
                        selection_after = self.selection_after_s
                    if requested_surface is not None:
                        reference = next(
                            (
                                s
                                for s in self.scan_memory.surfaces
                                if s.surface_id == requested_surface
                            ),
                            None,
                        )
                        selected_surfaces = (
                            []
                            if reference is None
                            or reference.observations[-1].semantic_id != self.identity
                            else [
                                s
                                for s in selected_surfaces
                                if registered_support(reference, s.observations[0])
                            ]
                        )
                    mask = self.sam.all_masks[ids.index(self.identity)]
                    options, _ = self.tracker.eligible_surfaces(selected_surfaces, sensor, mask)
                    geometric_choices = [
                        dict(
                            surface_id=s.surface_id,
                            sam3_id=self.identity,
                            static_surface_ids=[
                                old.surface_id
                                for old in self.scan_memory.surfaces
                                if old.observations[-1].semantic_id == self.identity
                                and registered_support(old, s.observations[0])
                            ],
                            frame=frame,
                            acquired_s=t,
                            pixel_bounds=np.argwhere(s.observations[0].mask())[:, ::-1]
                            .min(0)
                            .tolist()
                            + np.argwhere(s.observations[0].mask())[:, ::-1].max(0).tolist(),
                        )
                        for s in options
                    ]
                    if requested_pixel is not None:
                        u, v = requested_pixel
                        selected_surfaces = [
                            s
                            for s in options
                            if v < mask.shape[0]
                            and u < mask.shape[1]
                            and s.observations[0].mask()[v, u]
                        ]
                    else:
                        selected_surfaces = options
                    reference_state = (
                        "available"
                        if len(selected_surfaces) == 1
                        else "ambiguous_geometry"
                        if len(selected_surfaces) > 1
                        else "reference_unsupported"
                    )
                    can_start = t >= max(self.initialize_after, selection_after)
                    if can_start and self.tracker.initialize(
                        selected_surfaces,
                        sensor,
                        whole_leaf_mask=mask,
                        allow_ambiguous_seed=self.historical_selection,
                    ):
                        initial_candidates = self.tracker.candidates
                        native_result = self.engine.result
                        state = "initialized"
                    else:
                        state = (
                            self.tracker.diagnostics.get("seed_state", "insufficient_geometry")
                            if can_start
                            else "inspecting"
                        )
                geometry_s = time.perf_counter() - geometry_started
        else:
            reference_state = "initialized"
            mask = np.zeros(sensor["rgb"].shape[:2], bool)
            if present:
                mask = self.sam.all_masks[ids.index(self.identity)]
            self.engine.submit(sensor, [mask])
            native_result = self.engine.result
            state = "tracking" if present else "selected_identity_absent"
        result = dict(native_result or dict(objects=[]))
        native_s = 0 if native_result is None else native_result["latency_s"]
        result.update(
            frame=frame,
            capture_s=t,
            latency_s=time.perf_counter() - started,
            sam3=meta,
            cue=cue,
            semantic_absent=not present,
            identity=dict(
                sam3_id=self.identity,
                state=state,
                present=present,
                returned_same_id=returned,
                material_recovery_verified=False,
            ),
            geometric_choices=geometric_choices,
            geometry_candidates=geometry_candidates,
            initialization=dict(
                inspection_complete=t >= self.inspection_end,
                reference_state=reference_state,
                p2p_started=bool(self.tracker.candidates),
                selection_request=self.runtime.get("selection_request"),
            ),
            initial_candidates=initial_candidates,
            scan_memory=scan_snapshot,
            initialization_rejections=self.tracker.diagnostics.get("initialization_rejections", []),
            stages=dict(
                sam3_s=cue["latency_s"],
                geometry_and_seed_s=max(0, geometry_s - native_s),
                point2pose_s=native_s,
            ),
        )
        if self.resources:
            result["resources"] = dict(
                sample=self.resources.latest,
                sampled_peaks=self.resources.peaks.copy(),
                period_s=0.25,
            )
        binding_started = time.perf_counter()
        if initial_candidates:
            result["binding"] = [
                dict(
                    sam3_id=c.sam3_id,
                    geometric_object=c.surface.surface_id,
                    zone=c.candidate_id,
                    p2p_object_id=c.p2p_object_id,
                    reference_support=vars(c.geometry),
                    inspection_references=[
                        dict(
                            surface_id=old.surface_id,
                            support=vars(
                                surface_support(old, self.tracker.generation, self.config)
                            ),
                        )
                        for old in self.scan_memory.surfaces
                        if old.observations[-1].semantic_id == c.sam3_id
                        and registered_support(old, c.surface.observations[0])
                    ],
                )
                for c in initial_candidates
            ]
        result["stages"]["identity_binding_s"] = time.perf_counter() - binding_started
        result["latency_s"] = time.perf_counter() - started
        return result

    def close(self):
        # Kill workers before waiting on the coordinator thread; no old CUDA work survives.
        if self.resources is not None:
            self.resources.close()
        if self.sam is not None:
            self.sam.close()
        if self.native is not None:
            self.native.close()
        self.log.close()


class PublishedTrackingEngine:
    """The adapter consumes coordinator results; it never schedules a second inference."""

    def reset(self):
        pass


class SessionProvider:
    """Shared live/replay ingress with complete episode reset and immutable publications."""

    def __init__(
        self,
        recipe,
        engine,
        *,
        tracking=True,
        synchronous_simulation=False,
        inference_period_s=0.0,
        **tracking_options,
    ):
        self.binding, self.config, self.engine = recipe, recipe.config, engine
        if synchronous_simulation and (engine.replay or not callable(engine.clock)):
            raise ValueError("synchronous_simulation_requires_measured_clock")
        if (
            not np.isfinite(inference_period_s)
            or not 0 <= inference_period_s <= self.config["max_gap_s"]
        ):
            raise ValueError("invalid_geometry_inference_period")
        self.synchronous_simulation = synchronous_simulation
        self.inference_period_s = inference_period_s
        self.tracking = (
            PanelTracking(PublishedTrackingEngine(), self.config, **tracking_options)
            if tracking
            else None
        )
        self.generation = -1
        self.reset()

    @property
    def static(self):
        return self.scan_memory.surfaces

    @property
    def scan_state(self):
        return self.scan_memory.state

    def reset(self, *, preload=True):
        self.generation += 1
        self.engine.reset(preload=preload)
        if self.tracking:
            self.tracking.reset(self.generation)
        self.scan_memory = ScanMemory(self.config, self.generation)
        self.current_space = ObservedSpace(self.config, self.generation)
        self.scan_available = {}  # Bounded by the stationary inspection window.
        self.last_time = self.last_frame = self.last_estimate = self.last_cue = None
        self.last_publication = None
        self.last_submitted_s = None
        self.encoding = None
        self.diagnostics = dict(state="searching", semantic_calls=0)

    def select_object(self, sam3_id, geometric_object=None, *, reference_pixel=None, after_s=0.0):
        self.engine.prepare()
        self.engine.worker.select(
            sam3_id, geometric_object, reference_pixel=reference_pixel, after_s=after_s
        )

    def consume_results(self, now):
        event = self.engine.poll(now)
        if event is None:
            return False
        captured, _, result, ready = event
        result["publication_poll_wait_s"] = max(0.0, now - ready)
        ready = max(ready, now)
        publication_started = time.perf_counter()
        publication_base = ready
        if result["frame"] != int(captured["frame"]) or result["capture_s"] != float(
            captured["time_s"]
        ):
            raise ValueError("unsynchronized_session_publication")
        if result["scan_memory"] is not None:
            self.scan_memory = result["scan_memory"]
            self.scan_memory.generation = self.generation
            self.scan_memory.state.generation = self.generation
        result["cue"]["available_s"] = ready
        self.last_cue = captured, result["cue"]
        if self.tracking:
            if result["initial_candidates"]:
                self.tracking.candidates = result["initial_candidates"]
                self.tracking.diagnostics["binding"] = result["binding"]
            if result["objects"]:
                self.tracking.consume(
                    captured,
                    result,
                    ready,
                    clock=self.engine.clock if self.synchronous_simulation else None,
                )
                ready = self.tracking.diagnostics["available_s"]
            self.diagnostics["tracking"] = self.tracking.diagnostics
        self.current_space.views.clear()
        self.current_space.add(captured, ready)
        # New scene rays are separate from the frozen inspection surfaces/map.
        self.scan_memory.state.space = self.current_space
        self.diagnostics.update(
            state=result["identity"]["state"],
            identity=result["identity"],
            geometric_choices=result["geometric_choices"],
            initialization=result["initialization"],
            stages=result["stages"],
            semantic_calls=self.diagnostics["semantic_calls"] + 1,
            latency_s=result["latency_s"],
            masks=len(result["cue"]["masks"]),
            available_time_s=ready,
            result_age_s=now - float(captured["time_s"]),
            cue_frame=int(captured["frame"]),
            surfaces=len(self.static),
            object_reason=self.scan_state.reason,
            leaf_objects=len(self.scan_state.objects),
            fixed_surfaces=len(self.scan_state.fixed_surfaces),
            unresolved_surfaces=len(self.scan_state.unresolved_surfaces),
            queue=self.engine.queue_stats.copy(),
        )
        if result["scan_memory"] is not None:
            self.scan_available[float(captured["time_s"])] = ready
            # Worker snapshots contain acquisition-time estimates. Preserve actual
            # earlier publications when replacing the snapshot; never refresh old
            # inspection support with a later tracking acquisition.
            for surface in self.scan_memory.surfaces:
                for observation in surface.observations:
                    observation.available_s = self.scan_available.get(
                        observation.acquired_s, observation.available_s
                    )
            for obj in self.scan_memory.state.objects:
                obj.support = replace(
                    obj.support,
                    available_s=self.scan_available.get(
                        obj.support.supported_s, obj.support.available_s
                    ),
                )
        if self.tracking and result["initial_candidates"]:
            for candidate, binding in zip(self.tracking.candidates, result["binding"], strict=True):
                candidate.geometry = replace(candidate.geometry, available_s=ready)
                candidate.static_reference = replace(
                    candidate.static_reference, support=candidate.geometry
                )
                binding["reference_support"] = vars(candidate.geometry)
                for reference in binding["inspection_references"]:
                    support = reference["support"]
                    support["available_s"] = self.scan_available.get(
                        support["supported_s"], support["available_s"]
                    )
        self.diagnostics.update(
            inference_latency_s=result["latency_s"],
            queue_wait_s=result.get("queue_wait_s", 0),
            resources=result.get("resources"),
        )
        # Include the complete adapter, timestamp propagation and diagnostics work.
        publication_s = time.perf_counter() - publication_started
        ready = (
            max(publication_base, float(self.engine.clock()))
            if self.synchronous_simulation
            else publication_base + publication_s
        )
        if self.synchronous_simulation:
            result["publication_clock_domain"] = "simulation"
        result["provider_publication_s"] = publication_s
        result["publication_latency_s"] = ready - float(captured["time_s"])
        result["cue"]["available_s"] = ready
        if result["scan_memory"] is not None:
            self.scan_available[float(captured["time_s"])] = ready
            for surface in self.scan_memory.surfaces:
                for observation in surface.observations:
                    if observation.acquired_s == float(captured["time_s"]):
                        observation.available_s = ready
            for obj in self.scan_memory.state.objects:
                if obj.support.supported_s == float(captured["time_s"]):
                    obj.support = replace(obj.support, available_s=ready)
        for view in self.current_space.views.values():
            view["available_s"] = ready
        if self.tracking and result["objects"]:
            for candidate in self.tracking.candidates:
                if result["initial_candidates"]:
                    candidate.geometry = replace(candidate.geometry, available_s=ready)
                    candidate.static_reference = replace(
                        candidate.static_reference, support=candidate.geometry
                    )
                if candidate.state == "tracked" and candidate.support.supported_s == float(
                    captured["time_s"]
                ):
                    candidate.support = replace(candidate.support, available_s=ready)
            if result["initial_candidates"]:
                for candidate, binding in zip(
                    self.tracking.candidates, result["binding"], strict=True
                ):
                    binding["reference_support"] = vars(candidate.geometry)
            self.tracking.diagnostics.update(
                available_s=ready, latency_s=result["publication_latency_s"]
            )
        self.diagnostics.update(
            available_time_s=ready,
            latency_s=result["publication_latency_s"],
            publication_s=publication_s,
        )
        self.last_publication = captured, result
        return True

    def update(self, observation, *, now=None):
        from alexdoor_xas.perception.contracts import DoorEstimate
        from alexdoor_xas.recording.b1 import OBS_KEYS

        sensor = {
            key: (
                observation[key].detach().cpu().numpy()
                if hasattr(observation[key], "detach")
                else np.asarray(observation[key])
            )
            for key in OBS_KEYS
        }
        t, frame = float(sensor["time_s"]), int(sensor["frame"])
        now = (self.engine.clock() if self.engine.clock else t) if now is None else float(now)
        if np.isfinite(t) and (not np.isfinite(now) or now < t):
            raise ValueError("invalid_provider_clock")
        reason = None
        if not np.isfinite(t) or (
            self.last_time is not None and (t <= self.last_time or frame <= self.last_frame)
        ):
            reason = "nonmonotonic_observation"
        elif any(
            sensor[k].shape != (9,) or not np.isfinite(sensor[k]).all()
            for k in ("joint_position", "joint_velocity")
        ):
            reason = "invalid_proprioception"
        elif (
            sensor["camera_world"].shape != (4, 4)
            or sensor["intrinsics"].shape != (3, 3)
            or not np.isfinite(sensor["camera_world"]).all()
            or not np.isfinite(sensor["intrinsics"]).all()
        ):
            reason = "invalid_calibration"
        depth, valid = sensor["depth_m"], sensor["valid_depth"]
        if (
            depth.ndim != 3
            or depth.shape[-1] != 1
            or depth.shape != valid.shape
            or sensor["rgb"].shape != (*depth.shape[:2], 3)
        ):
            raise ValueError("Unaligned observed RGB-D")
        if not np.any(valid & np.isfinite(depth) & (depth > 0)) or not np.any(sensor["rgb"]):
            reason = reason or "missing_rgbd"
        if reason:
            self.reset(preload=False)
            return DoorEstimate(t, False, reason)
        # A capture gap does not fabricate intermediate SAM/P2P steps. Staleness
        # remains visible in support times; only explicit reset starts a new identity.
        self.last_time, self.last_frame = t, frame
        try:
            if self.synchronous_simulation and abs(now - t) > 1e-9:
                raise ValueError("synchronous_simulation_clock_mismatch")
            if (
                self.last_submitted_s is None
                or t - self.last_submitted_s >= self.inference_period_s - 1e-9
            ):
                self.engine.submit(sensor, start_s=now)
                self.last_submitted_s = t
                if self.synchronous_simulation:
                    self.engine.wait()
                    if abs(float(self.engine.clock()) - now) > 1e-9:
                        raise ValueError("synchronous_simulation_clock_advanced")
            self.consume_results(now)
        except (RuntimeError, EOFError, BrokenPipeError, ValueError) as error:
            self.engine.failed = True
            self.engine.pending = self.engine.latest = None
            self.diagnostics.update(state="session_failed", error=str(error))
            if self.tracking:
                for candidate in self.tracking.candidates:
                    candidate.state = "tracking_lost"
            else:
                raise
        local = self.tracking.local_state(now, sensor) if self.tracking else None
        self.last_estimate = DoorEstimate(
            t,
            False,
            "point2pose_local_diagnostic" if self.tracking else "static_scan_only",
            local=local,
        )
        return self.last_estimate

    def close(self):
        self.engine.close()


def session_provider(
    recipe,
    models,
    calibration,
    output,
    *,
    replay,
    tracking=True,
    selection=None,
    registration_diagnostics=False,
    clock=None,
    synchronous_simulation=False,
    inference_period_s=0.0,
    **options,
):
    from alexdoor_xas.perception.point2pose.runtime import Point2PoseEngine

    count = 0

    def factory(_):
        nonlocal count
        count += 1
        worker = PerceptionSession(
            models,
            recipe,
            calibration,
            Path(output) / f"session-{count}",
            tracking=tracking,
            generation=engine.generation - 1,
            registration_diagnostics=registration_diagnostics,
        )
        if selection is not None:
            worker.select(**selection)
        return worker

    engine = Point2PoseEngine(factory, replay=replay, capture_session=True, clock=clock)
    provider = SessionProvider(
        recipe,
        engine,
        tracking=tracking,
        synchronous_simulation=synchronous_simulation,
        inference_period_s=inference_period_s,
        calibration=calibration,
        **options,
    )
    try:
        engine.prepare()
    except BaseException:
        provider.close()
        raise
    (Path(output) / "startup.json").write_text(json.dumps(engine.worker.runtime, indent=2) + "\n")
    return provider
