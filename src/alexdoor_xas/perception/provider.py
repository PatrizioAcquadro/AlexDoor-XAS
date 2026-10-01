"""One causal observed-only geometry provider for replay and live sensor streams."""

import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from alexdoor_xas.action.frames import ObjectFrame, rot_z
from alexdoor_xas.perception.contracts import DoorEstimate, geometry_profile, validate_complete
from alexdoor_xas.perception.geometry import (
    contact_frame,
    dimensions_supported,
    fuse_surface,
    hinge_from_motion,
    matched_motion,
    observed_thickness,
    similar_surface,
    surfaces,
    track_surface,
    visible_hinge,
    voxel_points,
)
from alexdoor_xas.perception.inspection import load_inspection
from alexdoor_xas.perception.scan import ScanMemory
from alexdoor_xas.perception.tracking import PixelMotionTracker, moved_surface
from alexdoor_xas.perception.visual_worker import receive, send
from alexdoor_xas.recording.b1 import OBS_KEYS


@dataclass(frozen=True)
class PrototypeRecipe:
    """Diagnostic recipe, deliberately not a qualified PerceptionBinding release."""

    recipe_json: str

    def __post_init__(self):
        geometry_profile(self)
        if self.config.get("scan_fusion") not in (None, "object-v1"):
            raise ValueError("Unknown scan fusion recipe")

    @property
    def config(self):
        return json.loads(self.recipe_json)

    @property
    def obs_dim(self):
        return sum(self.config["visual_dims"]) + 18

    def to_dict(self):
        return dict(
            schema="b1.perception.prototype.v1",
            config=self.config,
            offline_passed=False,
            dynamic_passed=False,
            frozen=False,
        )


def load_recipe(path, root):
    config = json.loads(Path(path).read_text())
    config["inspection"] = load_inspection(Path(root) / config.pop("inspection_path"))
    return PrototypeRecipe(json.dumps(config, sort_keys=True, allow_nan=False))


class ModelWorker:
    """Private pipe to the isolated image-model process. Only RGB crosses this boundary."""

    def __init__(self, model_root, config, log=None):
        env = dict(os.environ, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", OMP_NUM_THREADS="4")
        self.process = subprocess.Popen(
            [sys.executable, "-m", "alexdoor_xas.perception.visual_worker", str(model_root)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=log,
            env=env,
        )
        send(self.process.stdin, config)
        response = receive(self.process.stdout)
        if response is None or "error" in response:
            self.close()
            raise RuntimeError(f"Visual worker initialization failed: {response}")
        self.runtime = response["ready"]

    def infer(self, rgb):
        started = time.perf_counter()
        send(self.process.stdin, dict(rgb=rgb.tobytes(), shape=rgb.shape))
        result = receive(self.process.stdout)
        if result is None or "error" in result:
            raise RuntimeError(f"Visual worker inference failed: {result}")
        result["model_latency_s"] = result["latency_s"]
        result["latency_s"] = time.perf_counter() - started
        return result

    def infer_video(self, images, prompts):
        send(self.process.stdin, dict(images=images, prompts=prompts))
        result = receive(self.process.stdout)
        if result is None or "error" in result:
            raise RuntimeError(f"Diagnostic video inference failed: {result}")
        return result

    def close(self):
        if self.process.stdin and not self.process.stdin.closed:
            self.process.stdin.close()
        try:
            self.process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait()
        if self.process.stdout:
            self.process.stdout.close()


class CueEngine:
    """Bounded one-in-flight inference with the same capture/completion event semantics.

    Replay waits for compute but releases its result only at capture + measured latency.
    Live computes in a background thread; no future/late result can rewrite an old output.
    """

    def __init__(self, worker, *, replay):
        self.worker, self.replay = worker, replay
        self.executor = ThreadPoolExecutor(max_workers=1) if not replay else None
        self.pending = None
        self.generation = 0

    def reset(self):
        self.generation += 1
        # A running CUDA request cannot be cancelled: it keeps the single slot until polled.

    def submit(self, sensor, view):
        if self.pending is not None:
            return False
        snapshot = {k: np.array(v, copy=True) for k, v in sensor.items()}
        request = (self.generation, snapshot, view)
        if self.replay:
            result = self.worker.infer(snapshot["rgb"])
        else:
            result = self.executor.submit(self.worker.infer, snapshot["rgb"])
        self.pending = request, result
        return True

    def poll(self, now):
        if self.pending is None:
            return None
        (generation, sensor, view), result = self.pending
        if not self.replay:
            if not result.done():
                return None
            result = result.result()
        ready = float(sensor["time_s"]) + result["latency_s"]
        if generation == self.generation and ready > now:
            return None
        self.pending = None
        if generation != self.generation:
            return None
        return sensor, view, result, ready

    def close(self):
        if self.executor:
            self.executor.shutdown(wait=True, cancel_futures=True)


class GeometryProvider:
    def __init__(self, binding, engine):
        self.binding, self.engine = binding, engine
        self.config = binding.config
        self.reset()

    @property
    def generation(self):
        return self.engine.generation

    def reset(self):
        self.engine.reset()
        self.last_estimate = None
        self.encoding = None
        self.last_time = None
        self.last_frame = None
        self.scan_index = 0
        self.next_semantic = 0.0
        self.static = []
        self.scan_memory = ScanMemory(self.config, self.generation)
        self.panel = self.closed = None
        self.current_candidates = []
        self.scene = np.empty((0, 3))
        self.motions = []
        self.pixel_tracker = PixelMotionTracker()
        self.hinge = self.thickness = None
        self.hinge_uncertainty = np.inf
        self.angle = 0.0
        self.stable_frames = 0
        self.lost_since = None
        self.diagnostics = dict(state="acquiring", semantic_calls=0, reacquisitions=0)
        self.last_cue = None

    @property
    def scan_state(self):
        return self.scan_memory.state

    def consume_results(self, now):
        """Release available static evidence without fabricating a sensor observation."""
        event = self.engine.poll(now)
        if event is None:
            return False
        captured, view, cue, ready = event
        self._fuse(captured, view, cue, available_s=ready)
        self.diagnostics["result_age_s"] = now - float(captured["time_s"])
        self.diagnostics["available_time_s"] = ready
        return True

    def due_view(self, t):
        times = self.config["inspection"]["sample_times_s"]
        if t <= times[-1] and t >= self.next_semantic:
            return self.last_frame
        return -1 if t >= self.next_semantic else None

    def _fuse(self, sensor, view, cue, *, available_s=None):
        cue = dict(
            cue,
            available_s=float(sensor["time_s"]) + cue["latency_s"]
            if available_s is None
            else available_s,
        )
        candidates = surfaces(cue, sensor, self.config)
        self.current_candidates = candidates
        self.last_cue = (sensor, cue)
        self.diagnostics.update(
            semantic_calls=self.diagnostics["semantic_calls"] + 1,
            latency_s=cue["latency_s"],
            masks=len(cue["masks"]),
            surfaces=len(candidates),
            cue_frame=int(sensor["frame"]),
        )
        if view >= 0:
            if self.config.get("scan_fusion") == "object-v1":
                self.scan_memory.add(candidates, sensor, view, available_s=cue["available_s"])
                self.static, self.scene = self.scan_memory.surfaces, self.scan_memory.scene
                state = self.scan_state
                self.diagnostics.update(
                    object_reason=state.reason,
                    leaf_objects=len(state.objects),
                    fixed_surfaces=len(state.fixed_surfaces),
                    unresolved_surfaces=len(state.unresolved_surfaces),
                )
            else:
                for patch in candidates:
                    if abs(patch.normal[2]) > 0.3:
                        continue
                    match = next(
                        (
                            i
                            for i, old in enumerate(self.static)
                            if similar_surface(old, patch, self.config)
                        ),
                        None,
                    )
                    if match is None:
                        patch.views.add(view)
                        self.static.append(patch)
                    else:
                        self.static[match] = fuse_surface(
                            self.static[match], patch, self.config, view
                        )
                self.scan_memory.space.add(sensor)
                from alexdoor_xas.perception.geometry import deproject

                depth, valid = sensor["depth_m"].squeeze(-1), sensor["valid_depth"].squeeze(-1)
                v, u = np.nonzero(
                    valid[::4, ::4] & np.isfinite(depth[::4, ::4]) & (depth[::4, ::4] > 0)
                )
                cloud = deproject(
                    depth[v * 4, u * 4],
                    np.c_[u * 4, v * 4],
                    sensor["intrinsics"],
                    sensor["camera_world"],
                )
                self.scene = voxel_points(np.r_[self.scene, cloud], self.config["voxel_m"], 60000)
            self._select_panel()
        elif self.closed is not None:
            matches = [
                p
                for p in candidates
                if abs(p.normal[2]) < 0.3
                and p.descriptor @ self.closed.descriptor >= self.config["descriptor_similarity"]
            ]
            if matches:
                reference = self.panel or self.closed
                matches.sort(key=lambda p: np.linalg.norm(p.center - reference.center))
                candidate = matches[0]
                # A plane's algebraic normal has two signs. Preserve its observed temporal
                # orientation instead of flipping when the camera crosses to the rear side.
                if candidate.normal @ reference.normal < 0:
                    candidate.normal *= -1
                    candidate.offset *= -1
                if candidate.normal @ reference.normal < np.cos(
                    np.deg2rad(self.config["association_angle_deg"])
                ):
                    recovery_motion = matched_motion(reference, candidate)
                    if recovery_motion is None:
                        self.panel = None
                        self.diagnostics["association_reason"] = "discontinuous_panel_motion"
                        return
                if len(matches) > 1 and np.linalg.norm(matches[1].center - candidate.center) < 0.03:
                    self.diagnostics["association_reason"] = "ambiguous_panel_jamb_wall"
                    self.panel = None
                    return
                motion = matched_motion(self.closed, candidate)
                if motion is not None:
                    self.motions.append(motion)
                    self.motions = self.motions[-100:]
                    hinge = hinge_from_motion(
                        self.motions,
                        np.deg2rad(self.config["motion_min_angle_deg"]),
                        self.config["floor_z_m"],
                    )
                    if hinge is not None:
                        self.hinge, self.hinge_uncertainty = hinge
                self.panel = candidate
                self.diagnostics["association_reason"] = "associated"
                thickness = observed_thickness(candidate, candidates, self.config)
                if thickness is not None:
                    self.thickness = thickness[0]

    def _select_panel(self):
        if self.config.get("scan_fusion") == "object-v1":
            self.scan_memory.surfaces = self.static
            state = self.scan_state
            # No area winner or arbitrary reference can resolve competing objects.
            self.closed = self.panel = None
            self.hinge, self.hinge_uncertainty = None, np.inf
            if len(state.objects) != 1 or state.reason != "supported_static_candidate":
                self.diagnostics["association_reason"] = "ambiguous_panel_jamb_wall"
                return
            obj = state.objects[0]
            self.closed = self.panel = obj.root
            self.diagnostics["association_reason"] = "associated"
            axes = [h.hypothesis for h in obj.hinges if h.hypothesis is not None]
            if len(axes) == 1:
                self.hinge = axes[0].frame.origin.copy()
                self.hinge_uncertainty = axes[0].support.position_bound_m
            return
        plausible = [
            s
            for s in self.static
            if len(s.views) >= 2
            and np.diff(s.bounds, axis=0)[0, 1] > 0.2
            and np.diff(s.bounds, axis=0)[0, 2] > 0.3
        ]

        def score(s):
            return len(s.views) * np.sqrt(np.prod(np.diff(s.bounds, axis=0)[0, 1:]))

        plausible.sort(key=score, reverse=True)
        self.diagnostics["static_hypotheses"] = len(plausible)
        if not plausible:
            return
        if len(plausible) > 1 and score(plausible[1]) >= self.config["ambiguity_ratio"] * score(
            plausible[0]
        ):
            self.panel = self.closed = None
            self.diagnostics["association_reason"] = "ambiguous_panel_jamb_wall"
            return
        self.closed = self.panel = plausible[0]
        self.diagnostics["association_reason"] = "associated"
        thickness = observed_thickness(self.panel, self.static, self.config)
        if thickness is not None:
            self.thickness = thickness[0]
        hinges = visible_hinge(self.panel, self.scene, self.config)
        if len(hinges) == 1:
            self.hinge, self.hinge_uncertainty = hinges[0], self.config["plane_tolerance_m"]

    def _estimate(self, t, tracked):
        cfg = self.config
        fields = dict(timestamp_s=t, valid=False, reason="unresolved_panel")
        failures = []
        if self.closed is None or tracked is None:
            reason = self.diagnostics.get("association_reason", "missing_panel_depth")
            if reason == "associated":
                reason = "missing_panel_depth"
            self.diagnostics["missing"] = [reason]
            return DoorEstimate(t, False, reason)
        closed, current = self.closed, tracked
        angle = float(
            np.arctan2(current.basis[1, 0], current.basis[0, 0])
            - np.arctan2(closed.basis[1, 0], closed.basis[0, 0])
        )
        angle = float(np.arctan2(np.sin(angle), np.cos(angle)))
        fields.update(panel_rotation=closed.basis @ rot_z(angle), signed_angle=angle)
        bounds = closed.bounds
        width, height = np.diff(bounds, axis=0)[0, 1:]
        fields["dimensions"] = np.array([width, height, self.thickness or np.nan])
        if not dimensions_supported(closed, cfg["position_uncertainty_m"]):
            failures.append("unobserved_panel_edges")
        if self.thickness is None:
            failures.append("unobserved_thickness")
        if self.hinge is None:
            failures.append("unresolved_hinge")
        else:
            frame = ObjectFrame(self.hinge.copy(), closed.basis.copy())
            fields["frame"] = frame
            hinge_y = float(self.hinge @ closed.basis[:, 1])
            distances = abs(bounds[:, 1] - hinge_y)
            side = -1 if distances[0] < distances[1] else 1
            contact_y = hinge_y - side * cfg["contact_fraction"] * width
            z = cfg["contact_height_m"]
            points = closed.points @ closed.basis
            local_support = (abs(points[:, 1] - contact_y) < 0.025) & (
                abs(points[:, 2] - z) < 0.025
            )
            if local_support.sum() < 12:
                failures.append("unobserved_contact_surface")
            else:
                from alexdoor_xas.perception.geometry import plane_fit

                patch = closed.points[local_support]
                fit = plane_fit(patch, cfg["plane_tolerance_m"], 12, np.random.default_rng(6100))
                if fit is None:
                    failures.append("unreliable_contact_surface")
                else:
                    normal, offset, _, _ = fit
                    if normal @ closed.normal < 0:
                        normal, offset = -normal, -offset
                    normal_local = normal @ closed.basis
                    x = (offset - normal_local[1] * contact_y - normal_local[2] * z) / normal_local[
                        0
                    ]
                    contact = np.array([x, contact_y, z]) @ closed.basis.T
                    rotation = rot_z(angle)
                    fields["contact_position"] = self.hinge + rotation @ (contact - self.hinge)
                    fields["contact_rotation"] = rotation @ contact_frame(normal)
            if self.hinge_uncertainty > cfg["position_uncertainty_m"]:
                failures.append("unreliable_hinge")
        if self.scan_index < len(cfg["inspection"]["sample_times_s"]):
            failures.append("incomplete_scan")
        if current.support_fraction < 0.5 or current.residual_m > cfg["plane_tolerance_m"]:
            failures.append("unreliable_surface_support")
        if (
            self.pixel_tracker.diagnostics.get("anchor_uncertainty_m", 0)
            > cfg["position_uncertainty_m"]
        ):
            failures.append("unreliable_motion_tracking")
        orientation_uncertainty = np.arctan2(current.residual_m, min(width, height))
        if orientation_uncertainty > np.deg2rad(cfg["rotation_uncertainty_deg"]):
            failures.append("unreliable_panel_orientation")
        if self.stable_frames < cfg["reacquisition_frames"]:
            failures.append("acquiring_identity")
        fields["confidence"] = min(current.support_fraction, max(0, 1 - current.residual_m / 0.02))
        fields["reason"] = failures[0] if failures else "observed"
        fields["valid"] = not failures
        estimate = DoorEstimate(**fields)
        if estimate.valid:
            validate_complete(estimate)
        self.diagnostics.update(
            missing=failures,
            measured_width_m=float(width),
            measured_height_m=float(height),
            measured_thickness_m=self.thickness,
            hinge_uncertainty_m=float(self.hinge_uncertainty),
            motion_fits=len(self.motions),
            observed_edges=closed.edges,
            orientation_uncertainty_deg=float(np.rad2deg(orientation_uncertainty)),
            angle_rad=angle,
        )
        return estimate

    def update(self, observation):
        self.last_estimate = self._update(observation)
        return self.last_estimate

    def _update(self, observation):
        sensor = {key: observation[key] for key in OBS_KEYS}
        sensor = {
            key: (value.detach().cpu().numpy() if hasattr(value, "detach") else np.asarray(value))
            for key, value in sensor.items()
        }
        t, frame = float(sensor["time_s"]), int(sensor["frame"])
        if not np.isfinite(t) or (
            self.last_time is not None and (t <= self.last_time or frame <= self.last_frame)
        ):
            self.reset()
            return DoorEstimate(t, False, "nonmonotonic_observation")
        if self.last_time is not None and t - self.last_time > self.config["max_gap_s"] + 1e-9:
            self.engine.reset()
            self.panel = None
            self.pixel_tracker.reset()
            self.stable_frames = 0
        self.last_time, self.last_frame = t, frame
        self.scan_index = sum(t >= sample for sample in self.config["inspection"]["sample_times_s"])
        if any(
            np.asarray(sensor[k]).shape != (9,) or not np.isfinite(sensor[k]).all()
            for k in ("joint_position", "joint_velocity")
        ):
            self.reset()
            return DoorEstimate(t, False, "invalid_proprioception")
        if (
            sensor["camera_world"].shape != (4, 4)
            or sensor["intrinsics"].shape != (3, 3)
            or not np.isfinite(sensor["camera_world"]).all()
            or not np.isfinite(sensor["intrinsics"]).all()
        ):
            self.encoding = None
            return DoorEstimate(t, False, "invalid_calibration")
        depth, valid = sensor["depth_m"], sensor["valid_depth"]
        if (
            depth.ndim != 3
            or depth.shape[-1] != 1
            or depth.shape != valid.shape
            or sensor["rgb"].shape != (*depth.shape[:2], 3)
        ):
            raise ValueError("Unaligned observed RGB-D")
        if not np.any(valid & np.isfinite(depth) & (depth > 0)) or not np.any(sensor["rgb"]):
            self.engine.reset()
            self.panel, self.encoding = None, None
            self.pixel_tracker.reset()
            self.stable_frames = 0
            self.lost_since = t if self.lost_since is None else self.lost_since
            self.diagnostics["state"] = "lost"
            self.diagnostics["missing"] = ["missing_rgbd"]
            return DoorEstimate(t, False, "missing_rgbd")
        self.consume_results(t)
        view = self.due_view(t)
        if view is not None and self.engine.submit(sensor, view):
            self.next_semantic = t + self.config["semantic_period_s"]
        if (
            self.scan_index == len(self.config["inspection"]["sample_times_s"])
            and self.closed is not None
        ):
            motion = self.pixel_tracker.update(sensor, self.closed)
            self.diagnostics.update(self.pixel_tracker.diagnostics)
            if motion is not None:
                self.panel = moved_surface(self.closed, motion)
                self.motions = (self.motions + [motion])[-100:]
                hinge = hinge_from_motion(
                    self.motions,
                    np.deg2rad(self.config["motion_min_angle_deg"]),
                    self.config["floor_z_m"],
                )
                if hinge is not None:
                    self.hinge, self.hinge_uncertainty = hinge
            self.diagnostics["hinge_motion_fits"] = sum(
                abs(np.arctan2(r[1, 0], r[0, 0])) >= np.deg2rad(self.config["motion_min_angle_deg"])
                and np.linalg.norm(r[:, 2] - [0, 0, 1]) <= 0.05
                for r, _, _ in self.motions
            )
        tracked = track_surface(self.panel, sensor, self.config) if self.panel is not None else None
        if tracked is None:
            if self.stable_frames > 0 and self.lost_since is None:
                self.lost_since = t
            self.stable_frames = 0
        else:
            self.stable_frames += 1
            if (
                self.lost_since is not None
                and self.stable_frames >= self.config["reacquisition_frames"]
            ):
                self.diagnostics.update(
                    reacquisitions=self.diagnostics["reacquisitions"] + 1,
                    recovery_s=t - self.lost_since,
                )
                self.lost_since = None
        estimate = self._estimate(t, tracked)
        self.diagnostics["state"] = "tracking" if estimate.valid else "ambiguous"
        if estimate.valid:
            self.encoding = np.r_[self.closed.descriptor, self.panel.descriptor]
        else:
            self.encoding = None
        return estimate
