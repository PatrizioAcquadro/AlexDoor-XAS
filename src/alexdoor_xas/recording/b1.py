"""Streaming causal B1 RGB-D episodes."""

import json
from pathlib import Path

import h5py
import numpy as np

from alexdoor_xas.assets.purdue import ARM_JOINTS
from alexdoor_xas.perception.contracts import RobotFeedback

SCHEMA = "b1.rgbd.v1"
PHASES = ("approach", "contact", "push", "hold", "release", "inspect")
OBS_KEYS = (
    "time_s",
    "frame",
    "rgb",
    "depth_m",
    "valid_depth",
    "joint_position",
    "joint_velocity",
    "camera_world",
    "intrinsics",
)


def robot_feedback_at(h5, row):
    """Legacy RGB-D replay has no torque channel, including when commands contain efforts."""
    if h5.attrs.get("schema") != SCHEMA:
        raise ValueError("Unsupported robot-feedback recording schema")
    return RobotFeedback.unavailable(float(h5["observations/time_s"][row]), ARM_JOINTS)


class B1Writer:
    def __init__(self, path, metadata, calibration):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.file = h5py.File(self.path, "x")
        self.file.attrs.update(schema=SCHEMA, complete=False)
        self.file.create_group("metadata").attrs.update(
            episode=json.dumps(metadata), calibration=json.dumps(calibration)
        )
        for name in ("observations", "commands", "annotations"):
            self.file.create_group(name)
        self.last_time = None
        self.last_frame = None
        self.count = 0

    def _append(self, group, values):
        for name, value in values.items():
            arr = np.asarray(value)
            if arr.dtype.kind in "OUS":
                raise ValueError("Step arrays must be numeric")
            key = f"{group}/{name}"
            if key not in self.file:
                # One image per chunk keeps memory bounded and supports random reads.
                self.file.create_dataset(
                    key,
                    shape=(0, *arr.shape),
                    maxshape=(None, *arr.shape),
                    dtype=arr.dtype,
                    chunks=(1, *arr.shape),
                    compression="lzf",
                )
            dataset = self.file[key]
            dataset.resize(len(dataset) + 1, axis=0)
            dataset[-1] = arr

    def observe(self, observation, annotation):
        if set(observation) != set(OBS_KEYS):
            raise ValueError("B1 observation keys differ from the observed-only contract")
        observation = dict(observation)
        observation["time_s"] = np.float64(observation["time_s"])
        observation["frame"] = np.int64(observation["frame"])
        t, frame = float(observation["time_s"]), int(observation["frame"])
        if self.last_time is None and abs(t) > 1e-9:
            raise ValueError("Initial observation must come from episode reset at t=0")
        if not np.isfinite(t) or (self.last_time is not None and t <= self.last_time):
            raise ValueError("Non-increasing observation timestamp")
        if self.last_frame is not None and frame <= self.last_frame:
            raise ValueError("Stale camera frame")
        rgb, depth, valid = (np.asarray(observation[k]) for k in ("rgb", "depth_m", "valid_depth"))
        near, far = self.calibration_interval
        expected = np.isfinite(depth) & (depth >= near) & (depth <= far) & (depth > 0)
        if rgb.shape != (*depth.shape[:2], 3) or depth.shape != valid.shape:
            raise ValueError("Unaligned RGB-D")
        if not np.array_equal(valid, expected):
            raise ValueError("Validity mask must depend only on metric depth")
        for key in ("joint_position", "joint_velocity"):
            value = np.asarray(observation[key])
            if value.shape != (9,) or not np.isfinite(value).all():
                raise ValueError("Expected nine finite observed joints")
        self._append("observations", observation)
        self._append("annotations", annotation)
        self.last_time, self.last_frame = t, frame
        self.count += 1

    @property
    def calibration_interval(self):
        return json.loads(self.file["metadata"].attrs["calibration"])["depth_interval_m"]

    def transition(self, command, observation, annotation):
        if self.count == 0 or not np.isclose(command["time_s"], self.last_time, atol=1e-9, rtol=0):
            raise ValueError("Command must follow its initial observation")
        self.observe(observation, annotation)
        self._append("commands", command)
        if self.count % 60 == 0:
            self.file.flush()

    def finish(self, outcome):
        if self.count < 2 or len(self.file["commands/time_s"]) != self.count - 1:
            raise ValueError("Incomplete causal episode")
        self.file["metadata"].attrs["outcome"] = json.dumps(outcome)
        self.file.attrs["complete"] = True
        self.file.flush()

    def close(self):
        self.file.close()


def validate_episode(path, *, images=True):
    """Reject unfinished/unsynchronized artifacts; return compact collection evidence."""
    with h5py.File(path, "r") as h5:
        if h5.attrs.get("schema") != SCHEMA or not h5.attrs.get("complete", False):
            raise ValueError(f"Incomplete or unsupported B1 episode: {path}")
        meta = json.loads(h5["metadata"].attrs["episode"])
        if meta["split"] not in ("train", "development"):
            raise ValueError("Test recordings cannot enter perception preparation")
        obs, commands, labels = (h5[k] for k in ("observations", "commands", "annotations"))
        n = len(obs["time_s"])
        if set(obs) != set(OBS_KEYS) or any(len(v) != n for v in obs.values()):
            raise ValueError("Observation lengths differ")
        if any(len(v) != n for v in labels.values()) or any(
            len(v) != n - 1 for v in commands.values()
        ):
            raise ValueError("Transition/annotation lengths differ")
        times = obs["time_s"][:]
        if len(times) < 2 or abs(times[0]) > 1e-9:
            raise ValueError("Episode must include reset and terminal observations")
        if not np.allclose(np.diff(times), meta["control_dt"], atol=1e-7):
            raise ValueError("Dropped or mis-timed control observation")
        if not np.array_equal(commands["time_s"][:], times[:-1]):
            raise ValueError("Noncausal command timestamps")
        if not (np.diff(obs["frame"][:]) > 0).all():
            raise ValueError("Repeated camera frames")
        if not np.isfinite(commands["joint_target"][:]).all():
            raise ValueError("Nonfinite applied command")
        means = []
        if images:
            near, far = json.loads(h5["metadata"].attrs["calibration"])["depth_interval_m"]
            for i in sorted(set((0, n // 2, n - 1))):
                d, v = obs["depth_m"][i], obs["valid_depth"][i]
                expected = np.isfinite(d) & (d > 0) & (d >= near) & (d <= far)
                if not np.array_equal(v, expected):
                    raise ValueError("Non-sensor validity mask")
                rgb = obs["rgb"][i]
                means.append(float(rgb.mean()))
                if not np.any(rgb):
                    raise ValueError("Empty RGB recording")
        outcome = json.loads(h5["metadata"].attrs["outcome"])
        if not outcome["passed"] or not outcome["released"] or outcome["hold_angle_deg"] is None:
            raise ValueError("Expert episode failed physical validity/hold/release")
        return dict(
            path=str(path),
            asset_id=meta["asset_id"],
            split=meta["split"],
            condition=meta["condition"],
            observations=n,
            duration_s=float(times[-1]),
            rgb_mean=means,
            outcome=outcome,
        )


def episode_paths(root, corpus, *, complete_campaign=True):
    entries = {e["asset_id"]: e for e in corpus["doors"] if e["split"] != "test"}
    paths = sorted(Path(root).glob("*/*/episode.hdf5"))
    found = set()
    for path in paths:
        validate_episode(path, images=False)
        with h5py.File(path, "r") as h5:
            meta = json.loads(h5["metadata"].attrs["episode"])
        entry = entries.get(meta["asset_id"])
        if entry is None or any(
            meta[k] != entry[k] for k in ("split", "handedness", "records_sha256")
        ):
            raise ValueError(f"Episode does not belong to the frozen corpus: {path}")
        key = (meta["asset_id"], meta["condition"])
        if key in found or meta["condition"] not in ("nominal", "light"):
            raise ValueError("Duplicate or unknown perception condition")
        found.add(key)
    if not paths:
        raise ValueError("No B1 recordings")
    expected = {(asset, condition) for asset in entries for condition in ("nominal", "light")}
    if complete_campaign and found != expected:
        raise ValueError(f"Incomplete perception campaign: {len(found)}/{len(expected)} episodes")
    return paths
