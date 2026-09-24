"""Shared validation for compact, self-contained learned-policy checkpoints."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

from alexdoor_xas.assets.identity import RobotAssetRef
from alexdoor_xas.dataset.loader import validate_obs_keys
from alexdoor_xas.dataset.normalize import DatasetNormStats
from alexdoor_xas.policies.common.training import torch_save_atomic

DATASET_FIELDS = ("task", "space", "version", "obs_keys", "view_id")
ACT_CHECKPOINT_FORMAT = "alexdoor_xas.act.v3"
DIFFUSION_CHECKPOINT_FORMAT = "alexdoor_xas.diffusion.v3"


@dataclass(frozen=True)
class CheckpointPayload:
    """Fields required to reconstruct a policy."""

    state_dict: Mapping[str, Any]
    obs_dim: int
    action_dim: int
    model_cfg: dict[str, Any]
    stats: DatasetNormStats
    robot_asset: RobotAssetRef | None


def _dataset_descriptor(config: Mapping[str, Any]) -> dict[str, Any]:
    source = config.get("dataset")
    if not isinstance(source, Mapping):
        raise ValueError("checkpoint config requires a dataset mapping")
    descriptor = {field: source.get(field) for field in DATASET_FIELDS}
    for field in ("task", "space", "version"):
        if not isinstance(descriptor[field], str) or not descriptor[field]:
            raise ValueError(f"checkpoint dataset {field} must be a non-empty string")
    descriptor["obs_keys"] = validate_obs_keys(descriptor["obs_keys"])
    view_id = descriptor["view_id"]
    if view_id is not None and (not isinstance(view_id, str) or not view_id):
        raise ValueError("checkpoint dataset view_id must be null or a non-empty string")
    return descriptor


def _validate_checkpoint_contract(
    *,
    dataset: Mapping[str, Any],
    stats: DatasetNormStats,
    obs_dim: int,
    action_dim: int,
    state_dict: Mapping[str, Any],
    robot_asset: RobotAssetRef | None,
) -> None:
    if obs_dim <= 0 or action_dim <= 0:
        raise ValueError("checkpoint dimensions must be positive")
    if dataset.get("space") != stats.action_space:
        raise ValueError("checkpoint action space does not match normalization stats")
    if dataset.get("obs_keys") != stats.obs_keys:
        raise ValueError("checkpoint observation keys do not match normalization stats")
    if dataset.get("view_id") != stats.view_id:
        raise ValueError("checkpoint dataset view does not match normalization stats")
    if stats.obs.dim != obs_dim or stats.action.dim != action_dim:
        raise ValueError("checkpoint dimensions do not match normalization stats")
    if not stats.train_episode_ids:
        raise ValueError("checkpoint normalization train split is empty")
    stats.action.validate()
    stats.obs.validate()
    if not isinstance(state_dict, Mapping) or not state_dict:
        raise ValueError("checkpoint state_dict must be a non-empty mapping")
    for name, value in state_dict.items():
        if not isinstance(value, torch.Tensor):
            raise ValueError(f"checkpoint weight {name!r} is not a tensor")
        if (value.is_floating_point() or value.is_complex()) and not torch.isfinite(value).all():
            raise ValueError(f"checkpoint weight {name!r} contains non-finite values")
    if robot_asset is None:
        raise ValueError("checkpoints require robot identity")


def _robot_asset_from_payload(payload: Any) -> RobotAssetRef | None:
    if payload is None:
        return None
    try:
        asset = RobotAssetRef.from_dict(payload)
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"invalid checkpoint robot identity: {error}") from error
    return asset


def save_checkpoint_payload(
    path: str | Path,
    checkpoint_format: str,
    model: Any,
    config: Mapping[str, Any],
    stats: DatasetNormStats,
    meta: Mapping[str, Any] | None = None,
    robot_asset: RobotAssetRef | None = None,
) -> Path:
    """Validate and atomically write the shared v3 checkpoint payload."""
    dataset = _dataset_descriptor(config)
    state_dict = model.state_dict()
    _validate_checkpoint_contract(
        dataset=dataset,
        stats=stats,
        obs_dim=model.obs_dim,
        action_dim=model.action_dim,
        state_dict=state_dict,
        robot_asset=robot_asset,
    )
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    torch_save_atomic(
        target,
        {
            "format": checkpoint_format,
            "state_dict": state_dict,
            "obs_dim": model.obs_dim,
            "action_dim": model.action_dim,
            "model_cfg": asdict(model.cfg),
            "dataset": dataset,
            "norm_stats": stats.to_dict(),
            "robot_asset": robot_asset.to_dict() if robot_asset is not None else None,
            "meta": dict(meta or {}),
        },
    )
    return target


def load_checkpoint_payload(
    path: str | Path,
    expected_format: str,
    checkpoint_label: str,
    map_location: str = "cpu",
) -> CheckpointPayload:
    """Load and validate model-neutral v3 checkpoint fields."""
    payload = torch.load(Path(path), map_location=map_location, weights_only=True)
    if not isinstance(payload, dict):
        raise ValueError(f"checkpoint {path} must contain a mapping")
    checkpoint_format = payload.get("format")
    if checkpoint_format != expected_format:
        raise ValueError(f"unsupported checkpoint format {checkpoint_format!r} in {path}")
    try:
        obs_dim = int(payload["obs_dim"])
        action_dim = int(payload["action_dim"])
        raw_model_cfg = payload["model_cfg"]
        if not isinstance(raw_model_cfg, Mapping):
            raise TypeError("model_cfg must be a mapping")
        model_cfg = dict(raw_model_cfg)
        state_dict = payload["state_dict"]
        dataset = _dataset_descriptor({"dataset": payload["dataset"]})
        stats = DatasetNormStats.from_dict(payload["norm_stats"])
        robot_asset = _robot_asset_from_payload(payload.get("robot_asset"))
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"invalid {checkpoint_label} checkpoint {path}: {error}") from error
    _validate_checkpoint_contract(
        dataset=dataset,
        stats=stats,
        obs_dim=obs_dim,
        action_dim=action_dim,
        state_dict=state_dict,
        robot_asset=robot_asset,
    )
    return CheckpointPayload(
        state_dict=state_dict,
        obs_dim=obs_dim,
        action_dim=action_dim,
        model_cfg=model_cfg,
        stats=stats,
        robot_asset=robot_asset,
    )
