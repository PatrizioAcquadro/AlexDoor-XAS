"""Frozen image cues and causal static RGB-D scan fusion."""

import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from alexdoor_xas.perception.inspection import load_inspection


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
    config["point2pose_selected"] = json.loads(
        (Path(root) / "configs/point2pose_selected.json").read_text()
    )
    return PrototypeRecipe(json.dumps(config, sort_keys=True, allow_nan=False))


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
