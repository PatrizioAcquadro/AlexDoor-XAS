"""Export matched A1-A4 datasets from one recorded episode set."""

from __future__ import annotations

import dataclasses
import json
import shutil
import subprocess
import tempfile
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from alexdoor_xas.action.spaces import (
    A1_JOINT_DELTA,
    A2_EE_DELTA,
    A3_OBJ_REL_EE_DELTA,
    A4_OBJ_CENTRIC_CHUNK,
)
from alexdoor_xas.dataset.loader import A4ChunkDataset, EpisodeDataset
from alexdoor_xas.dataset.robot_asset import dataset_robot_asset_payload
from alexdoor_xas.dataset.validate import (
    validate_a4_dataset,
    validate_dataset,
    validate_matched_action_space_datasets,
)
from alexdoor_xas.recording import EpisodeBuffer, write_episode


def export_datasets(
    episodes: list[EpisodeBuffer], datasets_root: str | Path, version: str = "v0"
) -> dict[str, Path]:
    """Validate and publish a new matched version without replacing existing data."""
    if not episodes:
        raise ValueError("cannot export an empty episode list")
    robot_asset = dataset_robot_asset_payload(episodes)
    task = episodes[0].meta.task
    for name in (task, version):
        if not name or Path(name).name != name or name in {".", ".."}:
            raise ValueError("task and version must be single directory names")
    ids = [episode.meta.episode_id for episode in episodes]
    if len(set(ids)) != len(ids):
        raise ValueError("episode ids must be unique")
    if any(episode.meta.action_space != A2_EE_DELTA for episode in episodes):
        raise ValueError("matched export requires recorded A2 actions")
    if any(episode.outcome is None for episode in episodes):
        raise ValueError("every exported episode requires an outcome")
    task_root = Path(datasets_root) / task
    spaces = [A2_EE_DELTA, A3_OBJ_REL_EE_DELTA, A4_OBJ_CENTRIC_CHUNK]
    joint_targets = [_has_joint_targets(episode) for episode in episodes]
    if any(joint_targets):
        if not all(joint_targets):
            raise ValueError("joint-target recording must be consistent across episodes")
        spaces.append(A1_JOINT_DELTA)
    exported = {space: task_root / space / version for space in spaces}
    for destination in exported.values():
        if destination.exists():
            raise FileExistsError(f"dataset version already exists: {destination}")
    try:
        relabeled = {
            A2_EE_DELTA: episodes,
            A3_OBJ_REL_EE_DELTA: [_relabel_to_door_frame(episode) for episode in episodes],
        }
        if A1_JOINT_DELTA in exported:
            relabeled[A1_JOINT_DELTA] = [_relabel_to_joint_delta(episode) for episode in episodes]
    except (KeyError, IndexError, ValueError) as error:
        raise ValueError(f"incomplete matched-action recording: {error}") from error

    task_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".export-", dir=task_root) as temporary:
        staging = Path(temporary)
        datasets = {}
        for space, buffers in relabeled.items():
            path = _export_hdf5(buffers, staging / space, robot_asset)
            datasets[space] = EpisodeDataset(path)
        a4 = A4ChunkDataset(_export_a4(episodes, staging / A4_OBJ_CENTRIC_CHUNK, robot_asset))
        checks = [validate_dataset(data, space) for space, data in datasets.items()]
        checks.extend(
            (validate_a4_dataset(a4), validate_matched_action_space_datasets(datasets, a4))
        )
        errors = [error for check in checks for error in check.errors]
        if errors:
            raise ValueError("invalid matched dataset: " + "; ".join(errors))
        published = []
        try:
            for space, destination in exported.items():
                destination.parent.mkdir(parents=True, exist_ok=True)
                if destination.exists():
                    raise FileExistsError(f"dataset version already exists: {destination}")
                (staging / space).rename(destination)
                published.append(destination)
        except BaseException:
            for destination in reversed(published):
                shutil.rmtree(destination)
            raise
    return exported


def _export_hdf5(
    episodes: list[EpisodeBuffer],
    directory: Path,
    robot_asset: dict | None,
) -> Path:
    directory.mkdir(parents=True)
    for episode in episodes:
        write_episode(episode, directory)
    _write_dataset_meta(episodes, directory, episodes[0].meta.action_space, robot_asset)
    return directory


def _export_a4(
    episodes: list[EpisodeBuffer],
    directory: Path,
    robot_asset: dict | None,
) -> Path:
    directory.mkdir(parents=True)
    lines = []
    for episode in episodes:
        meta = dataclasses.replace(episode.meta, action_space=A4_OBJ_CENTRIC_CHUNK)
        record = {
            "meta": meta.to_dict(),
            "chunks": episode.extras.get("a4_chunks", []),
            "outcome": episode.outcome.to_dict() if episode.outcome else None,
        }
        lines.append(json.dumps(record))
    (directory / "episodes.jsonl").write_text("\n".join(lines) + "\n")
    _write_dataset_meta(episodes, directory, A4_OBJ_CENTRIC_CHUNK, robot_asset)
    return directory


def _relabel_to_door_frame(episode: EpisodeBuffer) -> EpisodeBuffer:
    actions_door = np.asarray(episode.extras["action_door_frame"], dtype=np.float64)
    if actions_door.shape[0] != episode.n_steps:
        raise ValueError(
            f"episode {episode.meta.episode_id} has {episode.n_steps} steps but "
            f"{actions_door.shape[0]} door-frame actions"
        )
    steps = [
        dataclasses.replace(step, action=actions_door[i]) for i, step in enumerate(episode.steps)
    ]
    relabeled = EpisodeBuffer(
        meta=dataclasses.replace(episode.meta, action_space=A3_OBJ_REL_EE_DELTA),
        steps=steps,
        extras=dict(episode.extras),
    )
    relabeled.outcome = episode.outcome
    return relabeled


def _has_joint_targets(episode: EpisodeBuffer) -> bool:
    return bool(episode.steps) and "joint_pos_target" in episode.steps[0].proprio


def _relabel_to_joint_delta(episode: EpisodeBuffer) -> EpisodeBuffer:
    """Relabel targets as A1 deltas, including the recorded post-loop target."""
    targets = np.stack(
        [np.asarray(step.proprio["joint_pos_target"], dtype=np.float64) for step in episode.steps]
    )
    last = np.asarray(episode.extras["final_joint_pos_target"], dtype=np.float64).reshape(1, -1)
    deltas = np.diff(np.concatenate([targets, last], axis=0), axis=0)
    steps = [dataclasses.replace(step, action=deltas[i]) for i, step in enumerate(episode.steps)]
    relabeled = EpisodeBuffer(
        meta=dataclasses.replace(episode.meta, action_space=A1_JOINT_DELTA),
        steps=steps,
        extras=dict(episode.extras),
    )
    relabeled.outcome = episode.outcome
    return relabeled


def _write_dataset_meta(
    episodes: list[EpisodeBuffer],
    directory: Path,
    action_space: str,
    robot_asset: dict | None,
) -> None:
    outcomes = [episode.outcome for episode in episodes if episode.outcome is not None]
    meta = {
        "task": episodes[0].meta.task,
        "action_space": action_space,
        "n_episodes": len(episodes),
        "n_success": sum(1 for outcome in outcomes if outcome.success),
        "seeds": [episode.meta.seed for episode in episodes],
        "robot": episodes[0].meta.robot,
        "scene": episodes[0].meta.scene,
        "policy": episodes[0].meta.policy,
        "robot_asset": robot_asset,
        "git_commit": _git_commit(),
        "created_utc": datetime.now(UTC).isoformat(),
    }
    (directory / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")


def _git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[3],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return "unknown"
