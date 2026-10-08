"""Frozen image cues and causal static RGB-D scan fusion."""

import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from alexdoor_xas.perception.contracts import DoorEstimate
from alexdoor_xas.perception.geometry import surfaces
from alexdoor_xas.perception.inspection import load_inspection
from alexdoor_xas.perception.ipc import receive, send
from alexdoor_xas.perception.scan import ScanMemory
from alexdoor_xas.recording.b1 import OBS_KEYS


@dataclass(frozen=True)
class PrototypeRecipe:
    """Diagnostic recipe, deliberately not a qualified PerceptionBinding release."""

    recipe_json: str

    def __post_init__(self):
        if self.config.get("scan_fusion") != "object-v1":
            raise ValueError("Static scan requires the object-v1 recipe")

    @property
    def config(self):
        return json.loads(self.recipe_json)

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
        try:
            send(self.process.stdin, config)
            response = receive(self.process.stdout)
            if response is None or "error" in response:
                raise RuntimeError(f"Visual worker initialization failed: {response}")
        except BaseException:
            self.close()
            raise
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
            if "available_s" not in result:
                result["available_s"] = max(now, float(sensor["time_s"]) + result["latency_s"])
        result.setdefault("available_s", float(sensor["time_s"]) + result["latency_s"])
        ready = result.get("available_s", float(sensor["time_s"]) + result["latency_s"])
        if not np.isfinite(ready) or ready < float(sensor["time_s"]):
            self.pending = None
            raise ValueError("invalid_completion_time")
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
    """Static scan with optional Point2Pose local diagnostics; no qualified release."""

    def __init__(self, binding, engine, *, tracking=None):
        if binding.config.get("scan_fusion") != "object-v1":
            raise ValueError("Static scan requires the object-v1 recipe")
        self.binding, self.engine = binding, engine
        self.config = binding.config
        self.tracking = tracking
        self.reset()

    @property
    def generation(self):
        return self.episode_generation

    @property
    def static(self):
        return self.scan_memory.surfaces

    @property
    def scan_state(self):
        return self.scan_memory.state

    def reset(self):
        self.episode_generation = getattr(self, "episode_generation", -1) + 1
        if hasattr(self.engine, "restart"):
            self.engine.restart()
        else:
            self.engine.reset()
        if self.tracking is not None:
            self.tracking.reset(self.generation)
        self.last_estimate = self.encoding = None
        self.last_time = self.last_frame = None
        self.next_semantic = 0.0
        self.scan_memory = ScanMemory(self.config, self.generation)
        self.diagnostics = dict(state="static_scan", semantic_calls=0)
        self.last_cue = None

    def consume_results(self, now):
        """Release captured static evidence at completion, without a new observation."""
        event = self.engine.poll(now)
        if event is None:
            return False
        captured, view, cue, ready = event
        self._fuse(captured, view, cue, available_s=ready, started_s=now)
        self.diagnostics.update(
            result_age_s=now - float(captured["time_s"]), available_time_s=ready
        )
        return True

    def due_view(self, t):
        if self.tracking is not None and self.tracking.candidates:
            return None  # Native SAM2 already tracks the initialized hypotheses.
        if self.tracking is not None and t < self.config["inspection"]["sample_times_s"][0]:
            return None
        end = self.config["inspection"]["sample_times_s"][-1]
        return self.last_frame if t <= end and t >= self.next_semantic else None

    def _fuse(self, sensor, view, cue, *, available_s=None, started_s=None):
        cue = dict(
            cue,
            available_s=float(sensor["time_s"]) + cue["latency_s"]
            if available_s is None
            else available_s,
        )
        geometry_started = time.perf_counter()
        candidates = surfaces(cue, sensor, self.config)
        geometry_s = time.perf_counter() - geometry_started
        if (
            self.tracking is not None
            and float(sensor["time_s"]) >= self.config["inspection"]["sample_times_s"][0]
        ):
            # Use the common inspection's first completed view, not reset's
            # transient view. The original synchronized packet remains the seed.
            started_s = max(cue["available_s"], started_s or cue["available_s"]) + geometry_s
            self.tracking.initialize(candidates, sensor, start_s=started_s)
        self.last_cue = (sensor, cue)
        self.scan_memory.add(candidates, sensor, view, available_s=cue["available_s"])
        state = self.scan_state
        self.diagnostics.update(
            semantic_calls=self.diagnostics["semantic_calls"] + 1,
            latency_s=cue["latency_s"],
            masks=len(cue["masks"]),
            surfaces=len(candidates),
            cue_frame=int(sensor["frame"]),
            object_reason=state.reason,
            leaf_objects=len(state.objects),
            fixed_surfaces=len(state.fixed_surfaces),
            unresolved_surfaces=len(state.unresolved_surfaces),
        )

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
        self.last_time, self.last_frame = t, frame
        if any(
            sensor[k].shape != (9,) or not np.isfinite(sensor[k]).all()
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
            self.engine.reset()
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
            return DoorEstimate(t, False, "missing_rgbd")
        try:
            self.consume_results(t)
        except (RuntimeError, EOFError, BrokenPipeError, ValueError) as error:
            if self.tracking is None:
                raise
            self._tracking_failure(error)
        view = self.due_view(t)
        if view is not None and self.engine.submit(sensor, view):
            self.next_semantic = t + self.config["semantic_period_s"]
        if self.tracking is not None:
            try:
                local = self.tracking.update(sensor)
            except (RuntimeError, EOFError, BrokenPipeError, ValueError) as error:
                self._tracking_failure(error)
                local = self.tracking.local_state(t)
            self.diagnostics["tracking"] = self.tracking.diagnostics
            return DoorEstimate(t, False, "point2pose_local_diagnostic", local=local)
        return DoorEstimate(t, False, "static_scan_only")

    def _tracking_failure(self, error):
        state = "initialization_failed" if self.tracking.last_result is None else "tracking_failed"
        for candidate in self.tracking.candidates:
            candidate.state = state
        self.tracking.engine.failed = True
        self.tracking.engine.pending = self.tracking.engine.latest = None
        self.tracking.diagnostics.update(state=state, error=str(error))

    def close(self):
        if self.tracking is not None:
            self.tracking.engine.close()
        self.engine.close()
        if self.engine.worker is not None:
            self.engine.worker.close()
        if hasattr(self, "owned_log"):
            self.owned_log.close()
