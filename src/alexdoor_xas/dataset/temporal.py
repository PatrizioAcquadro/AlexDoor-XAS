"""Frame-indexed temporal preparation and lazy, bounded A1 training batches.

Raw sources are never modified. History and target horizon are reader views.
Eligibility is explicit prospective supervision, independent of target padding;
failed, ambiguous and terminal samples remain readable diagnostics.
"""

import json
from collections import OrderedDict
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F

from alexdoor_xas.assets.purdue import ARM_JOINTS
from alexdoor_xas.policies.temporal import (
    CausalObservationBuilder,
    gather,
    verified_feedback,
)

LAYOUT = "b1.temporal-frames.v1"


@torch.inference_mode()
def preprocess_rgbd(rgb, depth, valid, config, *, device="cuda:0"):
    """Same five channels and shared CNN baseline as rgbd-conv-v1; arbitrary size."""
    rgb, depth, valid = (torch.as_tensor(x, device=device) for x in (rgb, depth, valid))
    if (
        rgb.dtype != torch.uint8
        or valid.dtype != torch.bool
        or depth.shape != valid.shape
        or (depth.ndim != 3 or depth.shape[-1] != 1 or rgb.shape != (*depth.shape[:2], 3))
    ):
        raise ValueError("Invalid RGB-D shape/dtype")
    near, far = config.depth_interval_m
    # Validity is sensor evidence; do not infer missing depth as a measurement.
    if torch.any(valid & (~torch.isfinite(depth) | (depth < near) | (depth > far))):
        raise ValueError("Valid depth outside declared sensor interval")

    def pool(x):
        return F.adaptive_avg_pool2d(x, config.image_size)

    coverage = pool(valid.permute(2, 0, 1).float()[None])
    image = pool(rgb.permute(2, 0, 1).float()[None] / 255)
    metric = torch.where(valid, depth.clamp(max=config.depth_scale_m), 0)
    metric = pool(metric.permute(2, 0, 1).float()[None] / config.depth_scale_m)
    metric = metric / coverage.clamp_min(1e-8)
    return torch.cat((image, metric, coverage), dim=1)[0].cpu().numpy()


class CachedFrames:
    """Bounded deterministic preprocessed-frame cache, no trainable values."""

    def __init__(self, values, *, capacity=64):
        self.values, self.capacity = values, capacity
        self.cache = OrderedDict()
        self.reads = self.evictions = 0

    def gather_into(self, rows, output):
        """Read each distinct frame once into final caller-owned batch storage.

        Restore every duplicate in place; LRU capacity never grows with a batch.
        No expanded sliding-window cache or list of sample-sized images exists.
        """
        indices = np.asarray(rows).reshape(-1)
        destination = output.reshape(len(indices), *output.shape[-3:])
        for row in np.unique(indices):
            positions = np.flatnonzero(indices == row)
            if row < 0:
                destination[positions] = 0
                continue
            row = int(row)
            if row not in self.cache:
                self.cache[row] = np.asarray(self.values[row])
                self.reads += 1
                if len(self.cache) > self.capacity:
                    self.cache.popitem(last=False)
                    self.evictions += 1
            self.cache.move_to_end(row)
            destination[positions] = self.cache[row]

    def __getitem__(self, rows):
        rows = np.atleast_1d(rows)
        output = np.empty((len(rows), *self.values.shape[1:]), dtype=self.values.dtype)
        self.gather_into(rows, output)
        return output


class RawFrames:
    def __init__(self, source, rows, config, device):
        self.source, self.rows, self.config, self.device = source, np.asarray(rows), config, device
        self.reads = 0

    def __getitem__(self, rows):
        result = []
        for row in np.atleast_1d(rows):
            i = int(self.rows[int(row)])
            result.append(
                preprocess_rgbd(
                    *(self.source[key][i] for key in ("rgb", "depth_m", "valid_depth")),
                    self.config,
                    device=self.device,
                )
            )
            self.reads += 1
        return np.stack(result)


def _stream(acquired, values, *, available=None, generation=None, **extra):
    acquired = np.asarray(acquired, dtype="f8")
    return dict(
        acquired=acquired,
        available=acquired.copy() if available is None else np.asarray(available),
        generation=np.zeros(len(acquired), "i8") if generation is None else np.asarray(generation),
        values=values,
        **extra,
    )


def raw_streams(file, config, *, device="cuda:0", image_rows=None, legacy_synchronous=False):
    """Adapt declared native streams or explicitly synchronous legacy simulation.

    No final-test access. No evaluator, phase, truth or requested command enters
    streams. The caller must acknowledge the missing legacy arrival clock.
    """
    metadata = json.loads(file["metadata"].attrs["episode"])
    if metadata.get("split") not in ("train", "development"):
        raise ValueError("Only train/development recordings may be prepared")
    native = file.attrs.get("schema") == "b1.temporal-recording.v1"
    if native:
        image, state = file["captures"], file["control"]
        times, gen = state["acquired"][:], state["generation"][:]
        available = state["available"][:]
        image_times, image_gen, image_available = (
            image[key][:] for key in ("acquired", "generation", "available")
        )
    else:
        if not legacy_synchronous or file.attrs.get("schema") not in (
            "b1.rgbd.v1",
            "b1.policy_rollout.v1",
        ):
            raise ValueError(
                "Legacy data require explicit synchronous-simulation clock declaration"
            )
        image = file["observations"] if "observations" in file else file["captures"]
        state = file["observations"] if "observations" in file else file["control"]
        times = state["time_s"][:]
        gen = state["generation"][:] if "generation" in state else np.zeros(len(times), "i8")
        available = times.copy()
        image_times = image["time_s"][:]
        image_gen = (
            image["generation"][:]
            if "generation" in image
            else gen[np.searchsorted(times, image_times)]
        )
        image_available = image_times.copy()
    if image_rows is None and native:
        image_rows = np.arange(len(image_times), dtype="i8")
    if image_rows is None:
        # Native captures are already their declared independent cadence. Legacy
        # expert RGB-D at 60 Hz is viewed at 10 Hz, never deleted or rewritten.
        image_rows, previous, previous_gen = [], -np.inf, None
        for i, (t, g) in enumerate(zip(image_times, image_gen, strict=True)):
            if g != previous_gen or t - previous >= 0.1 - 1e-8:
                image_rows.append(i)
                previous, previous_gen = t, g
        image_rows = np.asarray(image_rows, "i8")
    else:
        image_rows = np.asarray(image_rows, "i8")
    frames = RawFrames(image, image_rows, config, device)
    streams = dict(
        images=_stream(
            image_times[image_rows],
            frames,
            available=image_available[image_rows],
            generation=image_gen[image_rows],
            source_index=image_rows,
        ),
        q=_stream(times, state["joint_position"], available=available, generation=gen),
        dq=_stream(times, state["joint_velocity"], available=available, generation=gen),
    )
    for field in ("q", "dq"):
        if native and f"{field}_valid" in state:
            streams[field]["valid"] = state[f"{field}_valid"][:]
    # Native measured feedback only with verified source/order/units/sign.
    torque = file.get("physics_feedback")
    supported = (
        torque is not None
        and "torque_nm" in torque
        and (
            (native and verified_feedback(json.loads(file["metadata"].attrs["feedback_contract"])))
            or (
                not native
                and torque.attrs.get("semantics") == "joint_output"
                and torque.attrs.get("source") == "simulated_actuator_feedback"
                and tuple(json.loads(torque.attrs.get("joint_names", "[]"))) == ARM_JOINTS
                and torque.attrs.get("units") == "N m"
                and torque.attrs.get("sign_convention") == "positive_joint_coordinate"
            )
        )
    )
    if supported:
        tau_t = torque["acquired"][:] if native else torque["time_s"][:]
        tau_g = (
            torque["generation"][:]
            if native
            else gen[np.minimum(np.searchsorted(times, tau_t), len(gen) - 1)]
        )
        streams["tau"] = _stream(
            tau_t,
            torque["torque_nm"],
            available=torque["available"][:] if native else torque["available_s"][:],
            generation=tau_g,
            valid=torque["healthy"][:] if native else np.ones(len(tau_t), bool),
        )
    else:
        streams["tau"] = _stream([], np.empty((0, 7)), valid=np.empty(0, bool))
    commands = file["commands"]
    if native:
        # A refusal/request is not an applied command and cannot overwrite carry-in.
        applied = (
            np.flatnonzero(commands["accepted"][:]) if "accepted" in commands else np.empty(0, "i8")
        )
        streams["target"] = _stream(
            commands["dispatch_s"][:][applied],
            commands["applied_target"][:][applied],
            available=commands["applied_s"][:][applied],
            generation=commands["generation"][:][applied],
        )
    else:
        ct = commands["time_s"][:]
        targets = commands["joint_target"][:]
        if targets.shape[1] != 7:
            names = metadata.get("joint_names")
            if names is None:
                raise ValueError("Full-robot command requires named joint mapping")
            targets = targets[:, [names.index(name) for name in ARM_JOINTS]]
        streams["target"] = _stream(ct, targets, generation=gen[np.searchsorted(times, ct)])
    return streams, metadata, times, gen


def prepare_temporal(
    source,
    destination,
    config,
    *,
    device="cuda:0",
    image_rows=None,
    legacy_synchronous=False,
    eligible=None,
    episode_id=None,
    image_compression=None,
):
    """Stream one frame at a time to an exclusive new store; never expand windows.

    eligible is an explicit command-row mask, default diagnostic-only. A chunk
    intersecting an ineligible real command is excluded as a whole, not padded.
    """
    if image_compression not in (None, "lzf", "gzip"):
        raise ValueError("Unsupported prepared image compression")
    image_storage = (
        {}
        if image_compression is None
        else dict(
            compression=image_compression,
            shuffle=True,
            **(dict(compression_opts=1) if image_compression == "gzip" else {}),
        )
    )
    with h5py.File(source, "r", rdcc_nbytes=1024**2) as raw, h5py.File(destination, "x") as out:
        out.attrs.update(
            layout=LAYOUT,
            complete=False,
            source=str(Path(source).resolve()),
            image_compression=image_compression or "none",
            preprocessing=json.dumps(
                dict(
                    image_size=config.image_size,
                    depth_scale_m=config.depth_scale_m,
                    depth_interval_m=config.depth_interval_m,
                )
            ),
            legacy_synchronous=legacy_synchronous,
        )
        out.attrs["source_complete"] = bool(raw.attrs.get("complete", False))
        out.attrs["source_outcome"] = raw["metadata"].attrs.get("outcome", "{}")
        out.attrs["source_terminal_support"] = bool(
            raw["metadata"].attrs.get("terminal_support", raw.attrs.get("complete", False))
        )
        streams, metadata, times, gen = raw_streams(
            raw, config, device=device, image_rows=image_rows, legacy_synchronous=legacy_synchronous
        )
        metadata = dict(metadata)
        if episode_id is not None:
            if not isinstance(episode_id, str) or not episode_id.strip():
                raise ValueError("Explicit derivative episode identity must be nonempty")
            metadata.setdefault("source_episode_id", metadata["episode_id"])
            metadata["preparation_source_episode_id"] = metadata["episode_id"]
            metadata["episode_id"] = episode_id
        if "feedback_contract" in raw["metadata"].attrs:
            out.attrs["source_feedback_contract"] = raw["metadata"].attrs["feedback_contract"]
        recorded_dt = metadata.get("control_dt", raw["metadata"].attrs.get("control_dt"))
        if recorded_dt is not None:
            metadata["control_dt"] = float(recorded_dt)
        out.attrs["episode"] = json.dumps(metadata)
        for key, stream in streams.items():
            group = out.create_group(key)
            for name, values in stream.items():
                if name == "values" and key == "images":
                    frame_shape = (5, *config.image_size)
                    column = group.create_dataset(
                        name,
                        shape=(len(stream["acquired"]), *frame_shape),
                        dtype="f4",
                        chunks=(1, *frame_shape),
                        **image_storage,
                    )
                    for i in range(len(column)):
                        column[i] = values[[i]][0]
                else:
                    group.create_dataset(name, data=np.asarray(values))
        # Preserve command-row targets separately, including refusals and holes.
        command = raw["commands"]
        if raw.attrs.get("schema") == "b1.temporal-recording.v1":
            ct, cg = command["state_acquired_s"][:], command["generation"][:]
            target = command["applied_target"][:]
            q_before = command["q_before"][:]
            accepted = command["accepted"][:]
        else:
            ct = command["time_s"][:]
            cg = gen[np.searchsorted(times, ct)]
            target = streams["target"]["values"]
            q_before = (
                raw["observations/joint_position"][: len(ct), :7]
                if "observations" in raw
                else raw["control/joint_position"][: len(ct), :7]
            )
            accepted = np.ones(len(ct), bool)
        valid = np.zeros(len(ct), bool) if eligible is None else np.asarray(eligible, bool)
        if valid.shape != ct.shape:
            raise ValueError("Eligibility must identify every command explicitly")
        actions = out.create_group("actions")
        for name, value in dict(
            time=ct, generation=cg, delta=target - q_before, eligible=valid & accepted
        ).items():
            actions.create_dataset(name, data=value)
        out.create_dataset("decision_time", data=times)
        out.create_dataset("decision_generation", data=gen)
        out.attrs["complete"] = True
        out.flush()
    return Path(destination)


def training_decisions(
    builder, times, generations, action_times, action_generations, eligibility, horizon=12
):
    """Shared ready command windows for native review and prepared sampling; no image reads."""
    if horizon < 1:
        raise ValueError("Prediction horizon must be positive")
    cfg = builder.config
    admitted = []
    for row, (t, gen) in enumerate(zip(times, generations, strict=True)):
        if t < cfg.static_time_s + cfg.warmup_s - 1e-9:
            continue
        commands = np.flatnonzero(action_generations == gen)
        position = np.searchsorted(action_times[commands], t - 1e-9)
        window = commands[position : position + horizon]
        if (
            not len(window)
            or not eligibility[window].all()
            or not np.allclose(
                action_times[window],
                t + np.arange(len(window)) * cfg.control_dt_s,
                atol=1e-8,
                rtol=0,
            )
        ):
            continue
        if builder.readiness(t, gen)["ready"]:
            admitted.append(row)
    return np.asarray(admitted, dtype="i8")


class TemporalEpisode:
    """Open one store; <=8 MiB retained numeric values and <=64 image frames."""

    def __init__(self, path, config, *, cache_frames=64):
        self.path, self.config = Path(path), config
        self.file = h5py.File(path, "r", rdcc_nbytes=1024**2)
        if self.file.attrs.get("layout") != LAYOUT or not self.file.attrs.get("complete"):
            self.close()
            raise ValueError("Incomplete temporal frame store")
        preprocessing = json.loads(self.file.attrs["preprocessing"])
        if preprocessing != dict(
            image_size=list(config.image_size),
            depth_scale_m=config.depth_scale_m,
            depth_interval_m=list(config.depth_interval_m),
        ):
            self.close()
            raise ValueError("Prepared preprocessing differs from observation recipe")
        self.metadata = json.loads(self.file.attrs["episode"])
        if self.metadata.get("split") not in ("train", "development"):
            self.close()
            raise ValueError("Final test remains sealed")
        recorded_config = self.metadata.get("observation_config")
        if recorded_config is not None and any(
            recorded_config[key] != getattr(config, key) for key in ("static_time_s", "warmup_s")
        ):
            self.close()
            raise ValueError("Reader startup metadata differs from the recorded episode")
        recorded_dt = self.metadata.get("control_dt")
        if recorded_dt is not None and not np.isclose(
            recorded_dt, config.control_dt_s, atol=1e-9, rtol=0
        ):
            self.close()
            raise ValueError("Action interval differs from the recorded control rate")
        streams = {}
        # Residual profiling shows repeated tiny HDF5 numeric gathers. Retain
        # at most 8 MiB of numeric values; larger stores stay lazy. Images never
        # enter this budget or materialize as a corpus.
        numeric_budget = 8 * 1024**2
        for key in ("images", "q", "dq", "tau", "target"):
            streams[key] = {
                name: value[:] if name != "values" else value
                for name, value in self.file[key].items()
            }
            values = streams[key]["values"]
            if key != "images" and values.nbytes <= numeric_budget:
                streams[key]["values"] = values[:]
                numeric_budget -= values.nbytes
        delta = self.file["actions/delta"]
        self.action_values = delta[:] if delta.nbytes <= numeric_budget else delta
        self.frames = CachedFrames(streams["images"]["values"], capacity=cache_frames)
        streams["images"]["values"] = self.frames
        self.builder = CausalObservationBuilder(
            config, streams, episode_id=self.metadata.get("episode_id", str(path))
        )
        self.times, self.generations = (
            self.file["decision_time"][:],
            self.file["decision_generation"][:],
        )
        self.action_times = self.file["actions/time"][:]
        self.action_generations = self.file["actions/generation"][:]
        self.eligibility = self.file["actions/eligible"][:]

    def sample(self, row, horizon=12):
        snapshot = self.builder.build(self.times[row], self.generations[row])
        return self._targets(row, horizon, snapshot)

    def _targets(self, row, horizon, snapshot):
        if horizon < 1:
            raise ValueError("Prediction horizon must be positive")
        t, gen = self.times[row], self.generations[row]
        rows = np.flatnonzero((self.action_generations == gen) & (self.action_times >= t - 1e-9))
        indices = np.full(horizon, -1, "i8")
        expected = t + np.arange(horizon) * self.config.control_dt_s
        # A gap inside a generation is missing supervision, never a repeated stop.
        for i, tick in enumerate(expected):
            match = rows[np.isclose(self.action_times[rows], tick, atol=1e-8, rtol=0)]
            if len(match):
                indices[i] = match[0]
        targets = gather(self.action_values, indices, (7,))
        present = indices >= 0
        eligible = bool(
            snapshot["ready"] and present.any() and self.eligibility[indices[present]].all()
        )
        # Only outside recorded command support is padding; internal holes reject.
        gen_commands = self.action_times[self.action_generations == gen]
        end = gen_commands.max() if len(gen_commands) else -np.inf
        padding = ~present & (expected > end + 1e-9)
        eligible &= bool(np.all(present | padding))
        return dict(**snapshot, actions=targets, is_pad=padding, supervision_eligible=eligible)

    def batch(self, rows, horizon=12, *, diagnostics=False):
        # Join each distinct decision once. Preserve caller order and duplicates.
        rows = np.asarray(rows, dtype="i8")
        if rows.ndim != 1 or not len(rows):
            raise ValueError("A batch requires a nonempty row vector")
        samples, indices = {}, {}
        for row in dict.fromkeys(rows.tolist()):
            snapshot, selected = self.builder.build_support(self.times[row], self.generations[row])
            samples[row] = self._targets(row, horizon, snapshot)
            indices[row] = np.where(snapshot["obs"]["image_valid"], selected, -1)
        if not diagnostics and not all(s["supervision_eligible"] for s in samples.values()):
            raise ValueError("Unready, ambiguous or failed supervision must remain diagnostic-only")
        ordered = [samples[row] for row in rows]
        obs = {key: np.stack([s["obs"][key] for s in ordered]) for key in ordered[0]["obs"]}
        obs["images"] = np.empty(
            (len(rows), self.config.image_count, 5, *self.config.image_size), "f4"
        )
        self.frames.gather_into(np.stack([indices[row] for row in rows]), obs["images"])
        return dict(
            obs=obs,
            actions=np.stack([s["actions"] for s in ordered]),
            is_pad=np.stack([s["is_pad"] for s in ordered]),
            episode_id=[self.builder.episode_id] * len(rows),
            decision_row=rows.copy(),
            decision_time_s=self.times[rows].copy(),
            generation=self.generations[rows].copy(),
        )

    def training_rows(self, horizon=12):
        """Admitted ready decision rows using numeric support only, no image reads.

        Sampler ordering/shuffling remains caller-owned. Full chunks crossing
        ineligible commands or internal holes are rejected; real tails are kept.
        """
        return training_decisions(
            self.builder,
            self.times,
            self.generations,
            self.action_times,
            self.action_generations,
            self.eligibility,
            horizon,
        )

    def batches(self, rows, *, batch_size=32, horizon=12, prefetch=True):
        from alexdoor_xas.policies.common.data import prefetch_batches

        def iterator():
            for start in range(0, len(rows), batch_size):
                yield self.batch(rows[start : start + batch_size], horizon)

        return prefetch_batches(iterator()) if prefetch else iterator()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def close(self):
        if getattr(self, "file", None):
            self.file.close()


def mixed_temporal_batch(selections, horizon=12):
    """Concatenate (episode, rows) selections, preserving order and audit identity.

    Startup times may differ; actual model semantics must match. No labels or
    audit metadata enter obs. Each episode retains its own causal join/masks.
    """
    selections = list(selections)
    if not selections:
        raise ValueError("A mixed batch requires episodes")
    first = selections[0][0]
    contract = (first.config.model_contract(), first.metadata.get("robot_asset"))
    identities = {}
    batches = []
    for episode, rows in selections:
        if (episode.config.model_contract(), episode.metadata.get("robot_asset")) != contract:
            raise ValueError("Mixed batch observation contracts differ")
        identity = episode.builder.episode_id
        if identity in identities and identities[identity] != episode.path.resolve():
            raise ValueError("Distinct recordings require unique episode identities")
        identities[identity] = episode.path.resolve()
        batches.append(episode.batch(rows, horizon))
    return dict(
        obs={key: np.concatenate([b["obs"][key] for b in batches]) for key in batches[0]["obs"]},
        episode_id=[identity for b in batches for identity in b["episode_id"]],
        **{
            key: np.concatenate([b[key] for b in batches])
            for key in ("actions", "is_pad", "decision_row", "decision_time_s", "generation")
        },
    )


def phase_focused_rows(episode, *, horizon=12, n_action_steps=6):
    """Valid source-row views near task events; annotations never enter obs.

    Select the chunk crossing positions (-H+1, -N, -1), the event and its first
    executed horizon (+1, +N). Stabilization also retains its real final tail.
    Overlapping event memberships remain one source episode, not new demos.
    No images are copied or additional prepared store created.
    """
    if not 0 < n_action_steps <= horizon:
        raise ValueError("Execution horizon must fit the prediction chunk")
    with h5py.File(episode.file.attrs["source"], "r") as raw:
        metadata = json.loads(raw["metadata"].attrs["episode"])
        if metadata.get("split") not in ("train", "development"):
            raise ValueError("Final test remains sealed")
        if raw.attrs.get("schema") != "b1.temporal-recording.v1":
            raise ValueError("Phase selection requires native source annotations")
        if not (
            np.array_equal(raw["commands/state_acquired_s"][:], episode.action_times)
            and np.array_equal(raw["commands/generation"][:], episode.action_generations)
            and metadata["episode_id"]
            == episode.metadata.get("preparation_source_episode_id", episode.builder.episode_id)
        ):
            raise ValueError("Phase annotations differ from the prepared source recording")
        names = metadata["phase_names"]
        phase = raw["evaluator/phase"][:]
        contact = raw["evaluator/contact_valid"][:]
    valid = episode.training_rows(horizon)
    events = {
        name: []
        for name in ("contact_acquisition", "contact_to_push", "hold_to_release", "stabilization")
    }
    if "contact" in names:
        acquired = np.flatnonzero((phase == names.index("contact")) & contact)
        if len(acquired):
            events["contact_acquisition"].append(int(acquired[0]))
    for i in np.flatnonzero(phase[1:] != phase[:-1]) + 1:
        pair = (names[int(phase[i - 1])], names[int(phase[i])])
        for event, target in (
            ("contact_to_push", ("contact", "push")),
            ("hold_to_release", ("hold", "release")),
            ("stabilization", ("release", "stabilize")),
        ):
            if pair == target:
                events[event].append(int(i))
    offsets = (-horizon + 1, -n_action_steps, -1, 0, 1, n_action_steps)
    selected = {}
    for name, command_rows in events.items():
        rows = []
        for command_row in command_rows:
            t = episode.action_times[command_row]
            gen = episode.action_generations[command_row]
            for offset in offsets:
                matches = valid[
                    (episode.generations[valid] == gen)
                    & np.isclose(
                        episode.times[valid],
                        t + offset * episode.config.control_dt_s,
                        atol=1e-8,
                        rtol=0,
                    )
                ]
                rows.extend(matches.tolist())
        if name == "stabilization" and "stabilize" in names:
            tail = np.flatnonzero(phase == names.index("stabilize"))
            if len(tail):
                t, gen = episode.action_times[tail[-1]], episode.action_generations[tail[-1]]
                rows.extend(
                    valid[
                        (episode.generations[valid] == gen)
                        & np.isclose(episode.times[valid], t, atol=1e-8, rtol=0)
                    ].tolist()
                )
        selected[name] = np.unique(np.asarray(rows, dtype="i8"))
    return selected
