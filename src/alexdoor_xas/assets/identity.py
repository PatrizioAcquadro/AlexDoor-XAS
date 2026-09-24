"""Robot-neutral identity used by recordings and checkpoints."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class RobotAssetRef:
    """Minimal robot identity embedded in every downstream artifact."""

    asset_id: str
    sha256: str

    def __post_init__(self) -> None:
        if not self.asset_id:
            raise ValueError("robot asset id must not be empty")
        if len(self.sha256) != 64 or any(c not in "0123456789abcdef" for c in self.sha256):
            raise ValueError("robot asset sha256 must be 64 lowercase hex characters")

    def to_dict(self) -> dict[str, str]:
        return {"id": self.asset_id, "sha256": self.sha256}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> RobotAssetRef:
        return cls(
            asset_id=str(value["id"]),
            sha256=str(value["sha256"]),
        )


def assert_checkpoint_runtime_compatible(checkpoint_asset, runtime_asset) -> str:
    """Reject weights recorded for a different robot asset."""
    if checkpoint_asset != runtime_asset:
        raise ValueError("checkpoint robot asset is incompatible with runtime asset")
    return "matching_asset"
