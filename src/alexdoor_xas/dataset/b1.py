"""Matched B1 feature/action episodes, separate from perception engineering data."""

import json
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np

from alexdoor_xas.action.b1 import (
    ACTION_DIMS,
    Segment,
    StageSequence,
    fit_segments,
    primitive_actions,
)
from alexdoor_xas.action.spaces import A4_OBJ_CENTRIC_CHUNK
from alexdoor_xas.assets.identity import RobotAssetRef
from alexdoor_xas.assets.purdue import ARM_JOINTS, NECK_JOINTS
from alexdoor_xas.policies.common.b1_contract import OBS_KEYS, PerceptionBinding, policy_contract
from alexdoor_xas.policies.observations import observation_columns, require_estimate

SCHEMA = "b1.policy-dataset.v1"
PURPOSE = "b1_matched_policy_demonstration"


@dataclass(frozen=True)
class MatchedEpisode:
    episode_id: str
    asset_id: str
    split: str
    times: np.ndarray
    observations: np.ndarray  # N+1, including the terminal observation
    actions: dict[str, np.ndarray]
    starts: dict[str, np.ndarray]  # indices in the shared physical episode


def compile_episode(
    *,
    episode_id,
    asset_id,
    asset_splits,
    observations,
    tools,
    joint_targets,
    goals,
    stages,
    binding,
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
    if (
        not n
        or len(observations) != n + 1
        or len(tools) != n + 1
        or len(joint_targets) != n
        or len(stages) != n
    ):
        raise ValueError("B1 requires N commands and N+1 observations/tool poses")
    times = np.asarray([o.time_s for o in observations])
    if not np.isfinite(times).all() or not (np.diff(times) > 0).all():
        raise ValueError("Noncausal B1 observation times")
    if not (np.diff([o.frame for o in observations]) > 0).all():
        raise ValueError("Repeated B1 camera frames")
    if not np.allclose(np.diff(times), times[1] - times[0], atol=1e-7, rtol=0):
        raise ValueError("Dropped B1 control observation")
    frames, angles, features, actions = [], [], [], {s: [] for s in ACTION_DIMS}
    for index, obs in enumerate(observations):
        if not obs.valid:
            raise ValueError(f"Unavailable manipulation observation at {index}: {obs.reason}")
        require_estimate(obs.estimate, obs.time_s, binding.config["max_gap_s"])
        columns = observation_columns(obs.features, binding)
        features.append(obs.features)
        frames.append(obs.estimate.frame)
        angles.append(obs.estimate.signed_angle)
        if index < n:
            rows = primitive_actions(
                columns["joint_position"][:7],
                joint_targets[index],
                tools[index],
                goals[index],
                obs.estimate.frame,
            )
            for space, row in rows.items():
                actions[space].append(row)
    starts = {s: np.arange(n, dtype=np.int64) for s in ACTION_DIMS}
    starts[A4_OBJ_CENTRIC_CHUNK], actions[A4_OBJ_CENTRIC_CHUNK] = fit_segments(
        tools, goals, frames, angles, stages
    )
    result = MatchedEpisode(
        episode_id,
        asset_id,
        split,
        times,
        np.stack(features),
        {s: np.stack(rows) for s, rows in actions.items()},
        starts,
    )
    validate_episode(result, binding, asset_splits)
    return result


def validate_episode(episode, binding, asset_splits):
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
    obs = np.asarray(episode.observations)
    if obs.shape != (n + 1, binding.obs_dim) or not np.isfinite(obs).all():
        raise ValueError("Invalid B1 features or missing terminal observation")
    if set(episode.actions) != set(ACTION_DIMS) or set(episode.starts) != set(ACTION_DIMS):
        raise ValueError("A matched B1 episode must contain A1-A4")
    for space, width in ACTION_DIMS.items():
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
    import torch

    from alexdoor_xas.recording.b1 import OBS_KEYS as SENSOR_KEYS
    from alexdoor_xas.recording.b1 import PHASES
    from alexdoor_xas.recording.b1 import validate_episode as validate_recording

    validate_recording(path, images=False)
    with h5py.File(path, "r") as source:
        metadata = json.loads(source["metadata"].attrs["episode"])
        if metadata.get("purpose") != PURPOSE:
            raise ValueError("Perception engineering recordings are not matched policy data")
        calibration = json.loads(source["metadata"].attrs["calibration"])
        if calibration["joint_names"] != list(ARM_JOINTS + NECK_JOINTS):
            raise ValueError("Recorded B1 joint order differs")
        if metadata.get("inspection") != observer.binding.config["inspection"]:
            raise ValueError("Recorded inspection differs from the frozen perception recipe")
        observer.reset()
        device = next(observer.estimator.estimator.parameters()).device
        values, tools = [], []
        obs = source["observations"]
        commands = source["commands"]
        phases = [PHASES[int(v)] for v in commands["phase"][:]]
        start = next((i for i, phase in enumerate(phases) if phase != "inspect"), len(phases))
        if start == len(phases) or "inspect" in phases[start:]:
            raise ValueError("Only a common initial inspection prefix may precede manipulation")
        for index in range(len(obs["time_s"])):
            sample = {key: torch.as_tensor(obs[key][index], device=device) for key in SENSOR_KEYS}
            encoded = observer.update(sample)
            if index >= start:
                values.append(encoded)
                tools.append(tool_fk(obs["joint_position"][index], calibration))
        from alexdoor_xas.action.frames import ObjectFrame

        goals = [
            ObjectFrame(p, r)
            for p, r in zip(
                commands["tool_position"][start:], commands["tool_rotation"][start:], strict=True
            )
        ]
        return compile_episode(
            episode_id=metadata["episode_id"],
            asset_id=metadata["asset_id"],
            asset_splits=asset_splits,
            observations=values,
            tools=tools,
            joint_targets=commands["joint_target"][start:],
            goals=goals,
            stages=phases[start:],
            binding=observer.binding,
        )


def export_dataset(episodes, destination, binding, robot_asset, asset_splits):
    """Publish one master with four aligned views; refuse existing destinations."""
    episodes = list(episodes)
    ids = [e.episode_id for e in episodes]
    if not ids or len(set(ids)) != len(ids):
        raise ValueError("B1 export needs distinct nonempty episode identities")
    for episode in episodes:
        validate_episode(episode, binding, asset_splits)
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".b1-", dir=destination.parent))
    try:
        metadata = dict(
            schema=SCHEMA,
            purpose=PURPOSE,
            perception=binding.to_dict(),
            robot_asset=robot_asset.to_dict(),
            asset_splits=asset_splits,
            episodes=[],
        )
        for index, episode in enumerate(episodes):
            name = f"episode_{index:06d}.hdf5"
            with h5py.File(staging / name, "x") as out:
                out.attrs.update(
                    episode_id=episode.episode_id, asset_id=episode.asset_id, split=episode.split
                )
                out.create_dataset("time_s", data=episode.times)
                out.create_dataset("observations", data=episode.observations)
                for space in ACTION_DIMS:
                    group = out.create_group(space)
                    group.create_dataset("actions", data=episode.actions[space])
                    group.create_dataset("starts", data=episode.starts[space])
            metadata["episodes"].append(name)
        (staging / "meta.json").write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n")
        staging.rename(destination)
    except BaseException:
        shutil.rmtree(staging)
        raise
    return destination


@dataclass(frozen=True)
class B1Record:
    episode_id: str
    asset_id: str
    split: str
    actions: np.ndarray
    obs: dict[str, np.ndarray]
    starts: np.ndarray
    times: np.ndarray
    terminal_observation: np.ndarray

    @property
    def n_steps(self):
        return len(self.actions)

    @property
    def action_dim(self):
        return self.actions.shape[1]


class B1Dataset:
    """ChunkSampler-compatible view, indexed by ticks for A1-A3, segments for A4."""

    def __init__(self, directory, space, *, binding=None):
        directory = Path(directory)
        metadata = json.loads((directory / "meta.json").read_text())
        if metadata.get("schema") != SCHEMA or metadata.get("purpose") != PURPOSE:
            raise ValueError("Unsupported B1 policy dataset")
        self.binding = PerceptionBinding.from_dict(metadata["perception"])
        if binding is not None and binding != self.binding:
            raise ValueError("B1 dataset perception mismatch")
        self.contract = policy_contract(self.binding, space)
        self.robot_asset = RobotAssetRef.from_dict(metadata["robot_asset"])
        self.action_space = space
        self.records = []
        for name in metadata["episodes"]:
            if Path(name).name != name:
                raise ValueError("Episode path must stay inside its dataset")
            with h5py.File(directory / name, "r") as source:
                episode = MatchedEpisode(
                    source.attrs["episode_id"],
                    source.attrs["asset_id"],
                    source.attrs["split"],
                    source["time_s"][:],
                    source["observations"][:],
                    {s: source[s + "/actions"][:] for s in ACTION_DIMS},
                    {s: source[s + "/starts"][:] for s in ACTION_DIMS},
                )
            validate_episode(episode, self.binding, metadata["asset_splits"])
            starts = episode.starts[space]
            columns = [
                observation_columns(row, self.binding) for row in episode.observations[starts]
            ]
            self.records.append(
                B1Record(
                    episode.episode_id,
                    episode.asset_id,
                    episode.split,
                    episode.actions[space],
                    {key: np.stack([c[key] for c in columns]) for key in OBS_KEYS},
                    starts,
                    episode.times[starts],
                    episode.observations[-1],
                )
            )
        if not self.records or len(set(self.episode_ids)) != len(self.records):
            raise ValueError("Empty or duplicate B1 episode membership")

    @property
    def action_dim(self):
        return ACTION_DIMS[self.action_space]

    @property
    def episode_ids(self):
        return [r.episode_id for r in self.records]

    def by_id(self, episode_id):
        return next(r for r in self.records if r.episode_id == episode_id)
