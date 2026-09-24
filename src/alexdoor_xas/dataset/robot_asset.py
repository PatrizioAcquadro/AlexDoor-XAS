"""Robot identity shared by dataset export and policy loading."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from alexdoor_xas.assets.identity import RobotAssetRef


def dataset_robot_asset_payload(episodes: list[Any]) -> dict[str, str] | None:
    if not episodes:
        raise ValueError("cannot derive robot identity from no episodes")
    if len({(item.meta.task, item.meta.robot) for item in episodes}) != 1:
        raise ValueError("dataset export cannot mix tasks or robots")
    refs = {(item.meta.robot_asset_id, item.meta.robot_asset_sha256) for item in episodes}
    if len(refs) != 1:
        raise ValueError("episodes do not share one robot asset identity")
    asset_id, sha256 = refs.pop()
    if not asset_id and not sha256:
        return None
    return RobotAssetRef(asset_id, sha256).to_dict()


def load_dataset_robot_asset(dataset_dir: str | Path) -> RobotAssetRef | None:
    meta_path = Path(dataset_dir) / "meta.json"
    try:
        payload = json.loads(meta_path.read_text()).get("robot_asset")
        return None if payload is None else RobotAssetRef.from_dict(payload)
    except (OSError, KeyError, TypeError, ValueError) as error:
        raise ValueError(f"invalid robot identity in {meta_path}: {error}") from error


def validate_dataset_episode_robot_asset(dataset: Any, ref: RobotAssetRef) -> None:
    for record in dataset.records:
        meta = record.meta
        if (meta.get("robot_asset_id"), meta.get("robot_asset_sha256")) != (
            ref.asset_id,
            ref.sha256,
        ):
            raise ValueError(
                f"episode {record.episode_id} robot asset identity differs from meta.json"
            )
