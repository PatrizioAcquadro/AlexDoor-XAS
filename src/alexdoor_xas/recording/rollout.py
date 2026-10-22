"""Lossless pilot rollout evidence with separate image, control and physics clocks."""

import json
from collections import deque
from pathlib import Path

import h5py
import numpy as np

from alexdoor_xas.recording.b1 import OBS_KEYS

SCHEMA = "b1.policy_rollout.v1"


def numpy(value):
    # Transfer image tensors only when retained, without importing a GPU stack.
    return value.detach().cpu().numpy() if hasattr(value, "detach") else np.asarray(value)


class RolloutWriter:
    """Retain consumed RGB-D once, all control states and actual submitted commands.

    Full-rate proprioception and source-frame joins reconstruct policy inputs
    without repeating the four-image tensor at every control tick. Evaluator
    telemetry has a separate write-only channel. A complete failed episode is
    valid evidence; completion is never a physical success certificate.
    """

    def __init__(self, path, metadata, calibration, *, control_dt, image_period_s):
        if not np.isfinite([control_dt, image_period_s]).all() or not (
            0 < control_dt <= image_period_s
        ):
            raise ValueError("invalid_rollout_recording_cadence")
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.file = h5py.File(self.path, "x")
        self.file.attrs.update(schema=SCHEMA, complete=False)
        self.file.create_group("metadata").attrs.update(
            episode=json.dumps(metadata),
            calibration=json.dumps(calibration),
            control_dt=float(control_dt),
            image_period_s=float(image_period_s),
        )
        for name in ("captures", "control", "commands", "physics_feedback", "evaluator"):
            self.file.create_group(name)
        # Keep bounded numeric chunk caches alive between row batches.
        self._columns = {name: {} for name in self.file if name != "metadata"}
        self._pending = {name: [] for name in self._columns if name != "captures"}
        self._write_failed = False
        self.dt, self.period = control_dt, image_period_s
        self.last_time = self.last_frame = self.last_capture_s = None
        self.captures = {}
        self.history = deque(maxlen=3)
        self.generation = None
        self.count = 0

    def append(self, group, values):
        """Numeric columns have stable shape; descriptions live in metadata/sidecars."""
        if self._write_failed:
            raise RuntimeError("rollout_write_failed")
        target = self._columns[group]
        # Validate the whole row before extending any existing column.
        for key, value in values.items():
            arr = np.asarray(value)
            if arr.dtype.kind not in "biuf" or (
                key in target
                and (target[key].shape[1:] != arr.shape or target[key].dtype != arr.dtype)
            ):
                raise ValueError("changed_rollout_column")
        if len(target) and set(target) != set(values):
            raise ValueError("changed_rollout_columns")
        for key, value in values.items():
            arr = np.asarray(value)
            if key not in target:
                image = group == "captures" and key in ("rgb", "depth_m", "valid_depth")
                chunks = (1, *arr.shape) if image else (128, *arr.shape)
                target[key] = self.file[group].create_dataset(
                    key,
                    shape=(0, *arr.shape),
                    maxshape=(None, *arr.shape),
                    dtype=arr.dtype,
                    chunks=chunks,
                    compression=None if image else "lzf",
                )
        if group in self._pending:
            self._pending[group].append(
                {key: np.array(value, copy=True) for key, value in values.items()}
            )
            if len(self._pending[group]) >= 128:
                self._flush_group(group)
        else:
            for key, value in values.items():
                column = target[key]
                column.resize(len(column) + 1, axis=0)
                column[-1] = value

    def _length(self, group):
        columns = self._columns[group]
        written = len(next(iter(columns.values()))) if columns else 0
        return written + len(self._pending.get(group, ()))

    def _value(self, group, key, index):
        column = self._columns[group][key]
        return (
            column[index] if index < len(column) else self._pending[group][index - len(column)][key]
        )

    def _flush_group(self, group):
        rows = self._pending[group]
        if not rows:
            return
        try:
            for key, column in self._columns[group].items():
                start = len(column)
                column.resize(start + len(rows), axis=0)
                column[start:] = np.stack([row[key] for row in rows])
        except BaseException:
            self._write_failed = True
            raise
        rows.clear()

    def flush(self):
        """Drain bounded numeric rows before the unchanged durable HDF5 flush."""
        if self._write_failed:
            raise RuntimeError("rollout_write_failed")
        for group in self._pending:
            self._flush_group(group)
        self.file.flush()

    def observe(self, sensor, observation, robot_state):
        """Call after the actual observer; its support identifies consumed frames.

        Image capture continues at the declared cadence even when the observer
        has no features, preserving failed inputs. Any additional actually used
        frame is retained too. No masks, NaNs or raw intrinsics are rewritten.
        """
        if set(sensor) != set(OBS_KEYS):
            raise ValueError("invalid_rollout_sensor_columns")
        t, frame = float(sensor["time_s"]), int(sensor["frame"])
        if (
            not np.isfinite(t)
            or (self.count == 0 and abs(t) > 1e-9)
            or self.last_time is not None
            and (
                not np.isclose(t - self.last_time, self.dt, atol=1e-8, rtol=0)
                or frame <= self.last_frame
            )
            or observation.time_s != t
            or observation.frame != frame
        ):
            raise ValueError("invalid_rollout_control_clock")
        support = observation.feature_support
        used = support is not None and support.recent.frame == frame
        if used and support.recent.acquired_s != t:
            raise ValueError("invalid_rollout_feature_clock")
        if (
            self.last_capture_s is None
            or t - self.last_capture_s >= self.period - 1e-9
            or used
            and frame not in self.captures
        ):
            raw = {key: numpy(value) for key, value in sensor.items()}
            raw.update(time_s=np.float64(t), frame=np.int64(frame))
            rgb, depth, valid = (np.asarray(raw[k]) for k in ("rgb", "depth_m", "valid_depth"))
            if (
                depth.ndim != 3
                or depth.shape[-1] != 1
                or rgb.shape != (*depth.shape[:2], 3)
                or valid.shape != depth.shape
                or rgb.dtype != np.uint8
                or valid.dtype != bool
            ):
                raise ValueError("unaligned_rollout_rgbd")
            self.append("captures", raw)
            self.captures[frame] = len(self.captures)
            self.last_capture_s = t
        indices = [-1] * 4
        acquired = available = np.full(2, np.nan, dtype=np.float64)
        if support is not None:
            generation = observation.generation
            if generation != self.generation:
                self.history.clear()
                self.generation = generation
            try:
                static = self.captures[support.static.frame]
                recent = self.captures[support.recent.frame]
            except KeyError as error:
                raise ValueError("unrecorded_policy_image") from error
            if not self.history or self.history[-1] != recent:
                self.history.append(recent)
            indices = [static] + [self.history[0]] * (3 - len(self.history)) + list(self.history)
            acquired = np.array([support.static.acquired_s, support.recent.acquired_s])
            available = np.array([support.static.available_s, support.recent.available_s])
            if (
                any(
                    block.acquired_s != self.file["captures/time_s"][index]
                    or block.generation != generation
                    for block, index in ((support.static, static), (support.recent, recent))
                )
                or not np.isfinite(np.r_[acquired, available]).all()
                or (acquired > available).any()
                or (available > t).any()
            ):
                raise ValueError("noncausal_recorded_policy_features")
        if set(robot_state) & {
            "time_s",
            "frame",
            "joint_position",
            "joint_velocity",
            "image_indices",
            "feature_acquired_s",
            "feature_available_s",
            "generation",
        }:
            raise ValueError("robot_state_overrides_policy_input")
        row = dict(
            time_s=np.float64(t),
            frame=np.int64(frame),
            joint_position=numpy(sensor["joint_position"]),
            joint_velocity=numpy(sensor["joint_velocity"]),
            image_indices=np.array(indices, dtype=np.int64),
            feature_acquired_s=acquired,
            feature_available_s=available,
            generation=np.int64(-1 if observation.generation is None else observation.generation),
            **robot_state,
        )
        self.append("control", row)
        self.last_time, self.last_frame = t, frame
        self.count += 1
        if self.count % 60 == 0:
            self.flush()

    def command(self, values):
        """Append the submitted command and actual bounded targets after execution."""
        index = self._length("commands")
        if index != self.count - 2 or not np.isclose(
            float(values["time_s"]), self._value("control", "time_s", index), atol=1e-9, rtol=0
        ):
            raise ValueError("noncausal_recorded_command")
        self.append("commands", values)

    def feedback(self, robot_rows, evaluator_rows):
        """Independent native streams remain separate from all observer inputs."""
        for group, rows in (("physics_feedback", robot_rows), ("evaluator", evaluator_rows)):
            for row in rows:
                timestamp = float(row["time_s"])
                length = self._length(group)
                if (
                    self.last_time is None
                    or not np.isfinite(timestamp)
                    or not 0 <= timestamp <= self.last_time
                    or length
                    and timestamp <= self._value(group, "time_s", length - 1)
                ):
                    raise ValueError("invalid_rollout_physics_clock")
                self.append(group, row)

    def finish(self, outcome):
        if not isinstance(outcome.get("passed"), bool) or not outcome.get("termination"):
            raise ValueError("missing_explicit_rollout_outcome")
        if self.count == 0 or self._length("commands") != self.count - 1:
            raise ValueError("incomplete_rollout_transitions")
        self.file["metadata"].attrs["outcome"] = json.dumps(outcome, allow_nan=False)
        self.flush()
        self.file.attrs["complete"] = True
        self.file.flush()

    def finish_unavailable(self, outcome):
        """Retain a failed initialization without inventing an initial observation."""
        if (
            outcome.get("passed") is not False
            or not outcome.get("termination")
            or outcome.get("learned_commands", 0) != 0
            or outcome.get("reference_commands", 0) != 0
            or self.count
            or any(self._length(group) for group in self._pending)
        ):
            raise ValueError("unavailable_rollout_requires_empty_failed_initialization")
        self.file["metadata"].attrs["outcome"] = json.dumps(outcome, allow_nan=False)
        self.file.attrs["initial_observation_unavailable"] = True
        self.flush()

    def close(self):
        if not self.file:
            return
        try:
            if not self._write_failed:
                self.flush()
        finally:
            self.file.close()
