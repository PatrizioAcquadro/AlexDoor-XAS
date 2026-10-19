"""One causal observation join for saved streams and bounded incremental ingress.

Absolute clocks, identities and source indices are audit support, never features.
Only the allowlisted measured fields below enter ACT. No supervision is accepted
by this builder. Applied targets are setpoints delivered to the controller, not
positions reached by the robot.
"""

from collections import deque
from dataclasses import asdict, dataclass

import numpy as np

TEMPORAL_RECIPE = "rgbd-finite-history-v1"
FIELDS = ("q", "dq", "tau", "target")
WIDTHS = (9, 9, 7, 7)


@dataclass(frozen=True)
class TemporalConfig:
    history_s: float = 4.0
    state_hz: float = 60.0
    image_size: tuple[int, int] = (120, 192)
    image_offsets_s: tuple[float, ...] | None = None
    static_time_s: float = 25.0
    warmup_s: float = 0.2
    image_gap_s: float = 0.15
    state_gap_s: float = 2 / 60
    depth_scale_m: float = 5.0
    depth_interval_m: tuple[float, float] = (0.1, 5.0)

    def __post_init__(self):
        for key in ("image_size", "depth_interval_m"):
            object.__setattr__(self, key, tuple(getattr(self, key)))
        if not np.isfinite(
            [
                self.history_s,
                self.state_hz,
                self.static_time_s,
                self.warmup_s,
                self.image_gap_s,
                self.state_gap_s,
                self.depth_scale_m,
                *self.depth_interval_m,
            ]
        ).all() or not (
            0 < self.history_s <= 60
            and 0 < self.state_hz <= 240
            and self.static_time_s >= 0
            and self.warmup_s >= 0
            and self.image_gap_s > 0
            and self.state_gap_s > 0
            and self.depth_scale_m > 0
            and 0 < self.depth_interval_m[0] < self.depth_interval_m[1]
            and len(self.image_size) == 2
            and all(type(x) is int and x > 0 for x in self.image_size)
        ):
            raise ValueError("Invalid temporal observation recipe")
        offsets = self.image_offsets_s
        if offsets is None:
            offsets = (0.0, 0.1, 0.2, *np.arange(0.4, self.history_s + 1e-8, 0.2))
            offsets = tuple(
                sorted(
                    {float(round(x, 9)) for x in offsets if x <= self.history_s + 1e-8},
                    reverse=True,
                )
            )
        else:
            offsets = tuple(float(x) for x in offsets)
        if (
            not offsets
            or not np.isfinite(offsets).all()
            or offsets[-1] != 0
            or any(x < 0 or x > self.history_s for x in offsets)
            or any(a <= b for a, b in zip(offsets[:-1], offsets[1:], strict=True))
        ):
            raise ValueError("Image offsets must be unique, oldest first, ending at zero")
        object.__setattr__(self, "image_offsets_s", offsets)

    @property
    def state_count(self):
        return int(round(self.history_s * self.state_hz)) + 1

    @property
    def image_count(self):
        return 1 + len(self.image_offsets_s)

    def to_dict(self):
        return asdict(self)


def validate_stream(stream):
    acquired, available, generation = (
        np.asarray(stream[k]) for k in ("acquired", "available", "generation")
    )
    if not (
        acquired.ndim == available.ndim == generation.ndim == 1
        and len(acquired) == len(available) == len(generation)
        and np.isfinite(acquired).all()
        and np.isfinite(available).all()
        and (acquired >= 0).all()
        and (available >= acquired).all()
    ):
        raise ValueError("Invalid stream clocks")
    for gen in np.unique(generation):
        if (np.diff(acquired[generation == gen]) < 0).any():
            raise ValueError("Stream clock regressed without reset")


def causal_indices(stream, slots, decision, generation, gap, *, strict=False):
    """Latest acquisition <= slot, available at decision; command support < slot."""
    a = np.asarray(stream["acquired"])
    v = np.asarray(stream["available"])
    g = np.asarray(stream["generation"])
    # Select small numeric support first. Never materialize a visual corpus.
    candidates = np.flatnonzero((g == generation) & (v <= decision) & (a <= decision))
    if not len(candidates):
        return np.full(len(slots), -1, dtype=np.int64)
    times = a[candidates]
    order = np.argsort(times, kind="stable")
    candidates, times = candidates[order], times[order]
    positions = np.searchsorted(times, slots, side="left" if strict else "right") - 1
    chosen = candidates[np.maximum(positions, 0)]
    valid = (positions >= 0) & (slots - a[chosen] <= gap + 1e-9)
    # Ack/dispatch availability must already hold at each historical state.
    if strict:
        for i, slot in enumerate(slots):
            eligible = candidates[(times < slot) & (v[candidates] <= slot)]
            if len(eligible):
                chosen[i], valid[i] = eligible[-1], True
            else:
                valid[i] = False
    return np.where(valid, chosen, -1)


def gather(values, indices, shape, dtype=np.float32):
    result = np.zeros((len(indices), *shape), dtype=dtype)
    valid = indices >= 0
    if valid.any():
        unique, inverse = np.unique(indices[valid], return_inverse=True)
        # h5py requires sorted unique row indices; duplicates restored afterwards.
        result[valid] = np.asarray(values[unique])[inverse]
    return result


class CausalObservationBuilder:
    def __init__(self, config, streams, *, episode_id):
        self.config, self.streams, self.episode_id = config, streams, str(episode_id)
        for stream in streams.values():
            validate_stream(stream)

    def build(self, decision_time_s, generation):
        cfg, t = self.config, float(decision_time_s)
        if not np.isfinite(t) or t < 0:
            raise ValueError("Invalid decision time")
        robot_slots = t - np.arange(cfg.state_count - 1, -1, -1) / cfg.state_hz
        image_slots = np.r_[min(t, cfg.static_time_s), t - np.array(cfg.image_offsets_s)]
        images = self.streams["images"]
        indices = causal_indices(images, image_slots, t, generation, cfg.image_gap_s)
        # Static reference only exists after its declared inspection endpoint.
        if t < cfg.static_time_s:
            indices[0] = -1
        values = gather(images["values"], indices, (5, *cfg.image_size))
        image_valid = indices >= 0
        image_times = np.zeros((len(indices), 2), np.float64)
        image_source = np.full(len(indices), -1, np.int64)
        for col, key in enumerate(("acquired", "available")):
            image_times[:, col] = gather(
                np.asarray(images[key])[:, None], indices, (1,), np.float64
            )[:, 0]
        image_source[image_valid] = np.asarray(images["source_index"])[indices[image_valid]]
        parts, masks, support = [], [], []
        for field, width in zip(FIELDS, WIDTHS, strict=True):
            stream = self.streams[field]
            idx = causal_indices(
                stream, robot_slots, t, generation, cfg.state_gap_s, strict=field == "target"
            )
            val = gather(stream["values"], idx, (width,))
            valid = idx >= 0
            if "valid" in stream:
                valid &= gather(np.asarray(stream["valid"])[:, None], idx, (1,), bool)[:, 0]
            valid &= np.isfinite(val).all(-1)
            val[~valid] = 0
            times = np.stack(
                [
                    gather(np.asarray(stream[key])[:, None], idx, (1,), np.float64)[:, 0]
                    for key in ("acquired", "available")
                ],
                axis=-1,
            )
            times[~valid] = 0
            parts.append(val)
            masks.append(valid)
            support.append(times)
        robot_valid = np.stack(masks, axis=-1)
        robot_times = np.stack(support, axis=1)
        # Relative age and availability delay only; invalid support contributes zero.
        robot_timing = np.stack(
            (t - robot_times[..., 0], robot_times[..., 1] - robot_times[..., 0]), axis=-1
        )
        robot_timing[~robot_valid] = 0
        image_timing = np.stack(
            (t - image_times[:, 0], image_times[:, 1] - image_times[:, 0]), axis=-1
        )
        image_timing[~image_valid] = 0
        ready = bool(
            t >= cfg.static_time_s + cfg.warmup_s - 1e-9
            and image_valid[0]
            and image_valid[-1]
            and robot_valid[-1, :2].all()
        )
        obs = dict(
            images=values,
            image_valid=image_valid,
            image_timing=image_timing.astype("f4"),
            robot=np.concatenate(parts, axis=-1),
            robot_valid=robot_valid,
            robot_timing=robot_timing.astype("f4"),
        )
        return dict(
            obs=obs,
            ready=ready,
            episode_id=self.episode_id,
            generation=int(generation),
            decision_time_s=t,
            image_source_index=image_source,
            image_times=image_times,
            robot_times=robot_times,
        )


class IncrementalObservationBuilder:
    """Bounded ingress of preprocessed frames and native numeric streams.

    Caller supplies explicit synchronized acquisition/availability clocks. Reset
    is mandatory for a new physical generation. Overflow fails visibly instead
    of silently losing events inside the requested window.
    """

    def __init__(self, config, *, episode_id, generation=0, max_image_hz=60, max_state_hz=240):
        self.config, self.episode_id = config, str(episode_id)
        self.limits = dict(
            images=int((config.history_s + config.image_gap_s) * max_image_hz) + 4,
            **{
                key: int((config.history_s + config.state_gap_s) * max_state_hz) + 4
                for key in FIELDS
            },
        )
        self.reset(generation)

    def reset(self, generation):
        self.generation = int(generation)
        self.rows = {key: deque() for key in ("images", *FIELDS)}
        self.static = None
        self.last_available = {key: -np.inf for key in self.rows}
        self.invalid = False

    def append(
        self,
        field,
        *,
        acquired,
        available,
        values,
        generation,
        source_index=-1,
        valid=True,
        clock_domain="episode_monotonic",
    ):
        if field not in self.rows or generation != self.generation:
            raise ValueError("Wrong stream or reset generation")
        if self.invalid:
            raise ValueError("Clock invalid until reset")
        if (
            clock_domain != "episode_monotonic"
            or not np.isfinite([acquired, available]).all()
            or not 0 <= acquired <= available
            or available < self.last_available[field]
        ):
            self.invalid = True
            raise ValueError("Unverified or regressed stream clock")
        row = dict(
            acquired=float(acquired),
            available=float(available),
            values=np.array(values, copy=True),
            generation=int(generation),
            source_index=int(source_index),
            valid=bool(valid),
        )
        self.last_available[field] = available
        rows = self.rows[field]
        rows.append(row)
        cutoff = (
            available
            - self.config.history_s
            - max(self.config.image_gap_s, self.config.state_gap_s)
        )
        # Keep one carry-in (especially the last controller target) and static.
        while len(rows) > 1 and rows[1]["acquired"] < cutoff:
            rows.popleft()
        if len(rows) > self.limits[field]:
            self.invalid = True
            raise ValueError("Temporal ingress capacity exceeded")
        if field == "images" and acquired <= self.config.static_time_s + 1e-9:
            if self.static is None or acquired >= self.static["acquired"]:
                self.static = row

    def build(self, decision_time_s):
        if self.invalid or any(v > decision_time_s for v in self.last_available.values()):
            raise ValueError("Invalid incremental snapshot clock; reset required")
        streams = {}
        for key, rows in self.rows.items():
            ordered = list(rows)
            if (
                key == "images"
                and self.static is not None
                and all(row is not self.static for row in ordered)
            ):
                ordered.append(self.static)
            ordered.sort(key=lambda row: row["acquired"])
            width = (
                (5, *self.config.image_size) if key == "images" else (WIDTHS[FIELDS.index(key)],)
            )
            streams[key] = {
                name: np.asarray([row[name] for row in ordered])
                for name in ("acquired", "available", "generation", "source_index", "valid")
            }
            streams[key]["values"] = (
                np.stack([row["values"] for row in ordered])
                if ordered
                else np.empty((0, *width), "f4")
            )
        return CausalObservationBuilder(self.config, streams, episode_id=self.episode_id).build(
            decision_time_s, self.generation
        )
