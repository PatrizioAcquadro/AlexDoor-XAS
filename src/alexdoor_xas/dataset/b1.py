"""Matched B1 episodes and explicit A1/A2 pilot data, separate from engineering data."""

import json
import shutil
import tempfile
from dataclasses import dataclass, replace
from pathlib import Path

import h5py
import numpy as np

from alexdoor_xas.action.b1 import (
    ACTION_DIMS,
    Segment,
    StageSequence,
    fit_segments,
    primitive_actions,
    robot_actions,
)
from alexdoor_xas.action.spaces import A1_JOINT_DELTA, A2_EE_DELTA, A4_OBJ_CENTRIC_CHUNK
from alexdoor_xas.assets.identity import RobotAssetRef
from alexdoor_xas.assets.purdue import ARM_JOINTS, NECK_JOINTS
from alexdoor_xas.dataset.storage import (
    FrameObservationWriter,
    ObservationArray,
    ObservationFiles,
    finite_observations,
)
from alexdoor_xas.perception.contracts import geometry_profile
from alexdoor_xas.policies.common.b1_contract import (
    OBS_KEYS,
    PilotBinding,
    observation_binding,
    policy_contract,
)
from alexdoor_xas.policies.observations import (
    observation_columns,
    require_estimate,
    require_features,
)

SCHEMA = "b1.policy-dataset.v1"
PURPOSE = "b1_matched_policy_demonstration"
PILOT_SCHEMA = "b1.policy-dataset.pilot.v1"
PILOT_PURPOSE = "b1_partial_policy_pilot"


def dataset_spaces(binding):
    return (
        (A1_JOINT_DELTA, A2_EE_DELTA) if isinstance(binding, PilotBinding) else tuple(ACTION_DIMS)
    )


@dataclass(frozen=True)
class MatchedEpisode:
    episode_id: str
    asset_id: str
    split: str
    times: np.ndarray
    observations: np.ndarray | ObservationArray  # N+1, including the terminal observation
    actions: dict[str, np.ndarray]
    starts: dict[str, np.ndarray]  # indices in the shared physical episode
    source_recording: str | None = None


def compile_episode(
    *,
    episode_id,
    asset_id,
    asset_splits,
    observations,
    tools,
    joint_targets,
    goals,
    stages=None,
    binding,
    observation_writer=None,
):
    """Compile an explicitly supplied complete physical episode without inference.

    Observation objects must already come from the shared causal builder. Teacher
    phase/commands are supervision; no annotation is concatenated into features.
    """
    if not isinstance(episode_id, str) or not episode_id:
        raise ValueError("Missing B1 episode identity")
    split = asset_splits.get(asset_id)
    if split not in ("train", "development"):
        raise ValueError("Only frozen train/development identities are admitted")
    n = len(goals)
    pilot = isinstance(binding, PilotBinding)
    if (
        not n
        or hasattr(observations, "__len__")
        and len(observations) != n + 1
        or len(tools) != n + 1
        or len(joint_targets) != n
        or not pilot
        and (stages is None or len(stages) != n)
    ):
        raise ValueError("B1 requires N commands and N+1 observations/tool poses")
    times, camera_frames, generations = [], [], set()
    spaces = dataset_spaces(binding)
    frames, angles, features, actions = [], [], [], {s: [] for s in spaces}
    for index, obs in enumerate(observations):
        if index > n:
            raise ValueError("B1 requires N commands and N+1 observations/tool poses")
        times.append(obs.time_s)
        camera_frames.append(obs.frame)
        generations.add(obs.generation)
        if obs.geometry_profile != geometry_profile(binding):
            raise ValueError("B1 observation geometry profile differs from its binding")
        require_features(obs, binding)
        if not pilot:
            if not obs.valid:
                raise ValueError(f"Unavailable manipulation observation at {index}: {obs.reason}")
            require_estimate(
                obs.estimate,
                obs.time_s,
                binding.config["max_gap_s"],
                profile=geometry_profile(binding),
                generation=obs.generation,
            )
        columns = observation_columns(obs.features, binding)
        if observation_writer is None:
            features.append(obs.features)
        else:
            observation_writer.append(obs.features)
        frames.append(obs.estimate.frame)
        angles.append(obs.estimate.signed_angle)
        if index < n:
            args = (
                columns["joint_position"][:7],
                joint_targets[index],
                tools[index],
                goals[index],
            )
            rows = robot_actions(*args) if pilot else primitive_actions(*args, obs.estimate.frame)
            for space, row in rows.items():
                actions[space].append(row)
    times = np.asarray(times)
    if len(times) != n + 1:
        raise ValueError("B1 requires N commands and N+1 observations/tool poses")
    if not np.isfinite(times).all() or not (np.diff(times) > 0).all():
        raise ValueError("Noncausal B1 observation times")
    if not (np.diff(camera_frames) > 0).all():
        raise ValueError("Repeated B1 camera frames")
    if pilot and len(generations) != 1:
        raise ValueError("Pilot episode crosses observation generations")
    starts = {s: np.arange(n, dtype=np.int64) for s in spaces}
    if not pilot:
        starts[A4_OBJ_CENTRIC_CHUNK], actions[A4_OBJ_CENTRIC_CHUNK] = fit_segments(
            tools, goals, frames, angles, stages
        )
    result = MatchedEpisode(
        episode_id,
        asset_id,
        split,
        times,
        np.stack(features) if observation_writer is None else observation_writer.finish(),
        {s: np.stack(rows) for s, rows in actions.items()},
        starts,
    )
    validate_episode(result, binding, asset_splits)
    return result


def validate_episode(episode, binding, asset_splits, *, check_observations=True):
    if (
        episode.split not in ("train", "development")
        or asset_splits.get(episode.asset_id) != episode.split
    ):
        raise ValueError("B1 episode does not match the frozen identity split")
    times = np.asarray(episode.times)
    n = len(times) - 1
    if n < 1 or not np.isfinite(times).all() or not (np.diff(times) > 0).all():
        raise ValueError("Invalid B1 physical time axis")
    if not np.allclose(np.diff(times), times[1] - times[0], atol=1e-7, rtol=0):
        raise ValueError("B1 control cadence is not uniform")
    if isinstance(binding, PilotBinding) and not np.allclose(
        np.diff(times), 1 / 60, atol=1e-7, rtol=0
    ):
        raise ValueError("Pilot control cadence must remain 60 Hz")
    obs = episode.observations
    if obs.shape != (n + 1, binding.obs_dim) or (
        check_observations and not finite_observations(obs)
    ):
        raise ValueError("Invalid B1 features or missing terminal observation")
    spaces = dataset_spaces(binding)
    if set(episode.actions) != set(spaces) or set(episode.starts) != set(spaces):
        raise ValueError("Pilot requires A1/A2; a matched B1 episode must contain A1-A4")
    for space in spaces:
        width = ACTION_DIMS[space]
        rows, starts = episode.actions[space], episode.starts[space]
        if rows.ndim != 2 or rows.shape[1] != width or not np.isfinite(rows).all():
            raise ValueError("Invalid B1 action tensor")
        if starts.dtype.kind not in "iu" or starts.shape != (len(rows),):
            raise ValueError("Invalid B1 action boundaries")
        if space != A4_OBJ_CENTRIC_CHUNK:
            if not np.array_equal(starts, np.arange(n)):
                raise ValueError("B1 primitive actions must cover every command")
            continue
        sequence, tick = StageSequence(), 0
        for start, row in zip(starts, rows, strict=True):
            if start != tick:
                raise ValueError("A4 segment boundaries leave a gap or overlap")
            segment = Segment.decode(row, remaining_ticks=n - tick)
            if not np.allclose(segment.encode(), row, atol=1e-7, rtol=0):
                raise ValueError("Training A4 labels must use canonical discrete encodings")
            sequence.accept(segment)
            tick += segment.ticks
            if segment.end_episode and tick != n:
                raise ValueError("A4 terminates before the physical episode ends")
        if tick != n or not sequence.ended:
            raise ValueError("A4 does not cover the complete physical episode")


def prepare_recording(path, observer, tool_fk, asset_splits):
    """Stream raw B1 observations through the same live builder.

    tool_fk(q, calibration) must use robot FK, including the frozen tool offset.
    Only an initial inspection prefix may be excluded from learned arm actions.
    Existing engineering recordings are deliberately not admitted here.
    """
    from alexdoor_xas.recording.b1 import OBS_KEYS as SENSOR_KEYS
    from alexdoor_xas.recording.b1 import PHASES
    from alexdoor_xas.recording.b1 import validate_episode as validate_recording

    validate_recording(path, images=False)
    with h5py.File(path, "r") as source:
        metadata = json.loads(source["metadata"].attrs["episode"])
        if metadata.get("purpose") not in (PURPOSE, PILOT_PURPOSE):
            raise ValueError("Perception engineering recordings are not policy data")
        purpose = PILOT_PURPOSE if isinstance(observer.binding, PilotBinding) else PURPOSE
        if metadata.get("purpose") != purpose:
            raise ValueError("Recording purpose differs from the policy binding")
        if metadata.get("split") != asset_splits.get(metadata["asset_id"]):
            raise ValueError("Recorded identity split differs from the frozen assignment")
        calibration = json.loads(source["metadata"].attrs["calibration"])
        encoding = observer.binding.config.get("encoding")
        if encoding and calibration["depth_interval_m"] != encoding["depth_interval_m"]:
            raise ValueError("Recorded depth interval differs from the encoder recipe")
        if calibration["joint_names"] != list(ARM_JOINTS + NECK_JOINTS):
            raise ValueError("Recorded B1 joint order differs")
        if metadata.get("inspection") != observer.binding.config["inspection"]:
            raise ValueError("Recorded inspection differs from the frozen perception recipe")
        observer.reset()
        obs = source["observations"]
        commands = source["commands"]
        phases = [PHASES[int(v)] for v in commands["phase"][:]]
        start = next((i for i, phase in enumerate(phases) if phase != "inspect"), len(phases))
        if start == len(phases) or "inspect" in phases[start:]:
            raise ValueError("Only a common initial inspection prefix may precede manipulation")

        def observations():
            for index in range(len(obs["time_s"])):
                sample = {key: obs[key][index] for key in SENSOR_KEYS}
                encoded = observer.update(sample)
                if index >= start:
                    yield encoded

        tools = [tool_fk(q, calibration) for q in obs["joint_position"][start:]]
        from alexdoor_xas.action.frames import ObjectFrame

        goals = [
            ObjectFrame(p, r)
            for p, r in zip(
                commands["tool_position"][start:], commands["tool_rotation"][start:], strict=True
            )
        ]
        writer = FrameObservationWriter(len(obs["time_s"]) - start, observer.binding)
        try:
            episode = compile_episode(
                episode_id=metadata["episode_id"],
                asset_id=metadata["asset_id"],
                asset_splits=asset_splits,
                observations=observations(),
                tools=tools,
                joint_targets=commands["joint_target"][start:],
                goals=goals,
                stages=phases[start:],
                binding=observer.binding,
                observation_writer=writer,
            )
        finally:
            writer.close()
        episode.observations.files.close()
        return replace(episode, source_recording=str(Path(path).resolve()))


def export_dataset(episodes, destination, binding, robot_asset, asset_splits, *, copy_raw=True):
    """Publish aligned views; optionally reference separately retained large raw files.

    The default remains self-contained. Reference mode keeps raw source identity
    explicit, and never requires privileged raw data to train or load the views.
    """
    ids = set()
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".b1-", dir=destination.parent))
    try:
        metadata = dict(
            schema=PILOT_SCHEMA if isinstance(binding, PilotBinding) else SCHEMA,
            purpose=PILOT_PURPOSE if isinstance(binding, PilotBinding) else PURPOSE,
            perception=binding.to_dict(),
            robot_asset=robot_asset.to_dict(),
            asset_splits=asset_splits,
            episodes=[],
        )
        for index, episode in enumerate(episodes):
            if not episode.episode_id or episode.episode_id in ids:
                raise ValueError("B1 export needs distinct nonempty episode identities")
            ids.add(episode.episode_id)
            validate_episode(episode, binding, asset_splits)
            name = f"episode_{index:06d}.hdf5"
            with h5py.File(staging / name, "x") as out:
                out.attrs.update(
                    episode_id=episode.episode_id, asset_id=episode.asset_id, split=episode.split
                )
                if episode.source_recording is not None:
                    with h5py.File(episode.source_recording, "r") as raw:
                        identity = json.loads(raw["metadata"].attrs["episode"])
                        if any(
                            identity.get(key) != getattr(episode, key)
                            for key in ("episode_id", "asset_id", "split")
                        ):
                            raise ValueError("Retained recording differs from the physical episode")
                        if copy_raw:
                            retained = out.create_group("raw")
                            retained.attrs.update(raw.attrs)
                            for key in raw:
                                raw.copy(key, retained)
                        else:
                            out.attrs["source_recording"] = str(
                                Path(episode.source_recording).resolve()
                            )
                            out.attrs["raw_retention"] = "separate_original_file"
                out.create_dataset("time_s", data=episode.times)
                values = episode.observations
                indexed = (
                    isinstance(values, ObservationArray)
                    and isinstance(values.rows, range)
                    and values.rows == range(values.source_shape[0])
                    and values.columns == range(values.source_shape[1])
                    and isinstance(values.files.get(values.path)["observations"], h5py.Group)
                )
                if indexed:
                    values.files.get(values.path).copy("observations", out)
                    values.files.close()
                else:
                    target = out.create_dataset(
                        "observations", shape=values.shape, dtype=values.dtype
                    )
                    for start in range(0, len(values), 64):
                        target[start : start + 64] = values[start : start + 64]
                for space in dataset_spaces(binding):
                    group = out.create_group(space)
                    group.create_dataset("actions", data=episode.actions[space])
                    group.create_dataset("starts", data=episode.starts[space])
            metadata["episodes"].append(name)
        if not ids:
            raise ValueError("B1 export needs distinct nonempty episode identities")
        (staging / "meta.json").write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n")
        staging.rename(destination)
    except BaseException:
        shutil.rmtree(staging)
        raise
    return destination


def index_dataset(source_directory, destination, *, copy_raw=False):
    """Non-destructive streaming conversion of prepared data to frame-indexed storage.

    Preserve metadata, normalization membership and numeric dtype. Embedded raw
    data can stay linked to the retained original file or be copied explicitly.
    Original files and their outcomes are never edited. No preprocessing is run.
    """
    source_directory, destination = Path(source_directory), Path(destination)
    metadata = json.loads((source_directory / "meta.json").read_text())
    binding = observation_binding(metadata["perception"])
    if destination.exists():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".b1-index-", dir=destination.parent))
    files = ObservationFiles()
    try:
        for name in metadata["episodes"]:
            if Path(name).name != name:
                raise ValueError("Episode path must stay inside its dataset")
            path = source_directory / name
            with h5py.File(path, "r") as source:
                values = ObservationArray.open(path, files, source["observations"])
                writer = FrameObservationWriter(len(values), binding, dtype=values.dtype)
                try:
                    for start in range(0, len(values), 8):
                        for row in values[start : start + 8]:
                            if not np.isfinite(row).all():
                                raise ValueError("Invalid prepared observation")
                            writer.append(row)
                    indexed = writer.finish()
                    with h5py.File(staging / name, "x") as out:
                        out.attrs.update(source.attrs)
                        for key in source:
                            if key == "observations":
                                continue
                            if key == "raw" and not copy_raw:
                                out[key] = h5py.ExternalLink(str(path.resolve()), "/raw")
                            else:
                                source.copy(key, out)
                        indexed.files.get(indexed.path).copy("observations", out)
                    indexed.files.close()
                    del indexed
                finally:
                    writer.close()
                del writer
            files.close()
        (staging / "meta.json").write_text(json.dumps(metadata, indent=2) + "\n")
        # Check complete contracts before publishing, without reconstructing a corpus.
        with B1Dataset(staging, dataset_spaces(binding)[0], check_observations=True):
            pass
        staging.rename(destination)
    except BaseException:
        shutil.rmtree(staging)
        raise
    finally:
        files.close()
    return destination


def materialize_train_data(sources, destination, *, normalization_episode_ids):
    """Stream exact float32 observations into contiguous train-only HDF5 files.

    A failed conversion leaves its staging directory for diagnosis. Original
    prepared files and raw recordings are never replaced or relabeled.
    """
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError(destination)
    if not sources:
        raise ValueError("Prepared training sources are required")
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".b1-materialize-", dir=destination.parent))
    metadata, ids = None, []
    for directory in map(Path, sources):
        source_meta = json.loads((directory / "meta.json").read_text())
        if source_meta.get("schema") != PILOT_SCHEMA or source_meta.get("purpose") != PILOT_PURPOSE:
            raise ValueError("Materialization requires prepared pilot data")
        if metadata is None:
            metadata = dict(source_meta, episodes=[], asset_splits={})
        for key in ("schema", "purpose", "perception", "robot_asset"):
            if source_meta[key] != metadata[key]:
                raise ValueError("Prepared training contracts differ")
        for name in source_meta["episodes"]:
            if Path(name).name != name:
                raise ValueError("Episode path escapes the prepared dataset")
            with h5py.File(directory / name, "r") as source:
                episode_id, asset_id = source.attrs["episode_id"], source.attrs["asset_id"]
                if (
                    source.attrs["split"] != "train"
                    or source_meta["asset_splits"].get(asset_id) != "train"
                ):
                    raise ValueError("Development/test data cannot enter train materialization")
                if episode_id in ids:
                    raise ValueError("Duplicate prepared training episode")
                name = f"episode_{len(ids):06d}.hdf5"
                with h5py.File(staging / name, "x") as out:
                    out.attrs.update(source.attrs)
                    files = ObservationFiles()
                    values = ObservationArray.open(source.filename, files, source["observations"])
                    target = out.create_dataset("observations", shape=values.shape, dtype="f4")
                    for start in range(0, len(values), 64):
                        block = values[start : start + 64]
                        converted = block.astype(np.float32)
                        if not np.isfinite(block).all() or not np.array_equal(
                            converted.astype(block.dtype), block
                        ):
                            raise ValueError("Observations are not exactly float32 representable")
                        target[start : start + 64] = converted
                    files.close()
                    for key in ("time_s", A1_JOINT_DELTA, A2_EE_DELTA):
                        source.copy(key, out)
                ids.append(episode_id)
                metadata["asset_splits"][asset_id] = "train"
                metadata["episodes"].append(name)
    fit_ids = list(normalization_episode_ids)
    if not fit_ids or len(set(fit_ids)) != len(fit_ids) or not set(fit_ids) <= set(ids):
        raise ValueError("Normalization requires unique declared train-only episodes")
    metadata["normalization_episode_ids"] = fit_ids
    (staging / "meta.json").write_text(json.dumps(metadata, indent=2) + "\n")
    staging.rename(destination)
    return destination


@dataclass(frozen=True)
class B1Record:
    episode_id: str
    asset_id: str
    split: str
    actions: np.ndarray
    obs: dict[str, np.ndarray | ObservationArray]
    starts: np.ndarray
    times: np.ndarray
    terminal_observation: np.ndarray
    observation_matrix: np.ndarray | ObservationArray | None = None

    @property
    def n_steps(self):
        return len(self.actions)

    @property
    def action_dim(self):
        return self.actions.shape[1]


class B1Dataset:
    """Selective views indexed by ticks for A1-A3, segments for A4.

    Structure/actions/clocks are checked at open. Selected observations are
    checked before sampling/normalization; check_observations=True additionally
    scans every observation at open using bounded memory. close() releases HDF5
    handles; subsequent reads reopen them. Small contiguous legacy arrays retain mmaps.
    """

    def __init__(self, directory, space, *, binding=None, check_observations=False):
        directory = Path(directory)
        self.dataset_dir = directory
        metadata = json.loads((directory / "meta.json").read_text())
        if metadata.get("schema") not in (SCHEMA, PILOT_SCHEMA):
            raise ValueError("Unsupported B1 policy dataset")
        self.binding = observation_binding(metadata["perception"])
        pilot = isinstance(self.binding, PilotBinding)
        if metadata.get("schema") != (PILOT_SCHEMA if pilot else SCHEMA) or metadata.get(
            "purpose"
        ) != (PILOT_PURPOSE if pilot else PURPOSE):
            raise ValueError("Unsupported B1 policy dataset")
        spaces = dataset_spaces(self.binding)
        if space not in spaces:
            raise ValueError(
                "Representation unavailable in this dataset; pilot contains A1/A2 only"
            )
        if binding is not None and binding != self.binding:
            raise ValueError("B1 dataset perception mismatch")
        self.contract = policy_contract(self.binding, space)
        self.robot_asset = RobotAssetRef.from_dict(metadata["robot_asset"])
        self.action_space = space
        self.records = []
        self._files = ObservationFiles()
        for name in metadata["episodes"]:
            if Path(name).name != name:
                raise ValueError("Episode path must stay inside its dataset")
            with h5py.File(directory / name, "r") as source:
                values = source["observations"]
                # Validation reads bounded blocks; create the mmap afterwards so
                # scanning does not populate the entire mapping's resident set.
                lazy = ObservationArray.open(directory / name, self._files, values)
                offset = (
                    values.id.get_offset()
                    if isinstance(values, h5py.Dataset) and values.file == source
                    else None
                )
                observations = lazy
                episode = MatchedEpisode(
                    source.attrs["episode_id"],
                    source.attrs["asset_id"],
                    source.attrs["split"],
                    source["time_s"][:],
                    observations,
                    {s: source[s + "/actions"][:] for s in spaces},
                    {s: source[s + "/starts"][:] for s in spaces},
                )
            validate_episode(
                episode,
                self.binding,
                metadata["asset_splits"],
                check_observations=check_observations,
            )
            if offset is not None and np.prod(lazy.shape) * lazy.dtype.itemsize <= 8 * 1024**2:
                observations = np.memmap(
                    directory / name,
                    mode="r",
                    dtype=lazy.dtype,
                    offset=offset,
                    shape=lazy.shape,
                )
            starts = episode.starts[space]
            consecutive = np.array_equal(starts, np.arange(len(starts)))
            if isinstance(observations, ObservationArray):
                matrix = observations.view(rows=slice(0, len(starts)) if consecutive else starts)
            else:
                matrix = observations[: len(starts)] if consecutive else lazy.view(rows=starts)
            widths = [len(v) for v in observation_columns(matrix[0], self.binding).values()]
            offsets = np.cumsum([0, *widths])
            columns = {
                key: (
                    matrix.view(columns=slice(begin, end))
                    if isinstance(matrix, ObservationArray)
                    else matrix[:, begin:end]
                )
                for key, begin, end in zip(OBS_KEYS, offsets[:-1], offsets[1:], strict=True)
            }
            terminal = observations[-1]
            if not np.isfinite(terminal).all():
                raise ValueError("Invalid B1 terminal observation")
            self.records.append(
                B1Record(
                    episode.episode_id,
                    episode.asset_id,
                    episode.split,
                    episode.actions[space],
                    columns,
                    starts,
                    episode.times[starts],
                    terminal,
                    matrix,
                )
            )
        if not self.records or len(set(self.episode_ids)) != len(self.records):
            raise ValueError("Empty or duplicate B1 episode membership")
        train = tuple(r.episode_id for r in self.records if r.split == "train")
        self.normalization_episode_ids = tuple(metadata.get("normalization_episode_ids", train))
        if "normalization_episode_ids" in metadata and (
            not self.normalization_episode_ids
            or len(set(self.normalization_episode_ids)) != len(self.normalization_episode_ids)
            or not set(self.normalization_episode_ids) <= set(train)
        ):
            raise ValueError("Normalization requires unique declared train-only episodes")

    @property
    def action_dim(self):
        return ACTION_DIMS[self.action_space]

    @property
    def episode_ids(self):
        return [r.episode_id for r in self.records]

    def by_id(self, episode_id):
        return next(r for r in self.records if r.episode_id == episode_id)

    def close(self):
        self._files.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
