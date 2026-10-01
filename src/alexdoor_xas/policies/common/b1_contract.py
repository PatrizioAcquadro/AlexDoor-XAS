"""Explicit B1 compatibility and the still-pending Phase 6.0 release boundary."""

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path

from alexdoor_xas.action.b1 import ACTION_DIMS, ACTION_SCHEMA
from alexdoor_xas.assets.purdue import ARM_JOINTS, NECK_JOINTS
from alexdoor_xas.perception.contracts import LEGACY_FULL_STATE

OBS_KEYS = ("rgbd_static", "rgbd_recent", "joint_position", "joint_velocity")
OBS_SCHEMA = "b1.policy-observation.v1"
CONTRACT_SCHEMA = "b1.policy-contract.v1"
RELEASE_SCHEMA = "b1.perception.release.v2"


def check_sha(value):
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(c not in "0123456789abcdef" for c in value)
    ):
        raise ValueError("Expected a SHA256 artifact reference")


@dataclass(frozen=True)
class PerceptionBinding:
    """A release declaration, not a mechanism for qualifying an estimator.

    Phase 6.0 must supply the evidence-backed release after its offline/dynamic
    gates and freeze. Existing confidence-qualified checkpoints alone do not pass.
    JSON storage keeps caller mutation from changing a bound recipe in place.
    """

    release_json: str

    def __post_init__(self):
        release = self.to_dict()
        if release.get("schema") != RELEASE_SCHEMA or any(
            release.get(key) is not True for key in ("offline_passed", "dynamic_passed", "frozen")
        ):
            raise ValueError("B1 requires qualified and frozen Phase 6.0 perception")
        artifacts = release.get("artifacts")
        if (
            not isinstance(artifacts, dict)
            or not artifacts
            or any(not isinstance(name, str) or not name for name in artifacts)
        ):
            raise ValueError("Perception requires named artifact identities")
        for digest in artifacts.values():
            check_sha(digest)
        cfg = release.get("config", {})
        if any(value != LEGACY_FULL_STATE for value in (
            cfg.get("geometry_profile", LEGACY_FULL_STATE),
            release.get("geometry_profile", LEGACY_FULL_STATE),
        )):
            raise ValueError("Perception release v2 only supports legacy-full-state")
        dims = cfg.get("visual_dims")
        if (
            not isinstance(dims, list)
            or len(dims) != 2
            or any(type(v) is not int or v < 1 for v in dims)
            or not cfg.get("inspection")
        ):
            raise ValueError("Incompatible B1 perception recipe")
        for key, positive in (("max_gap_s", True), ("warmup_s", False)):
            value = cfg.get(key)
            if (
                type(value) not in (int, float)
                or not math.isfinite(value)
                or value < 0
                or positive
                and value == 0
            ):
                raise ValueError("Incompatible B1 perception timing")
        from alexdoor_xas.perception.inspection import validate_inspection

        validate_inspection(cfg["inspection"])

    def verify_artifacts(self, paths):
        """Verify exact release inputs before a provider loads them; this is not qualification."""
        expected = self.to_dict()["artifacts"]
        if set(paths) != set(expected):
            raise ValueError("Perception artifact names differ from the release")
        for name, path in paths.items():
            with Path(path).open("rb") as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
            if digest != expected[name]:
                raise ValueError(f"Frozen perception artifact mismatch: {name}")

    @classmethod
    def from_dict(cls, release):
        return cls(json.dumps(release, sort_keys=True, allow_nan=False))

    def to_dict(self):
        return json.loads(self.release_json)

    @property
    def config(self):
        return self.to_dict()["config"]

    @property
    def obs_dim(self):
        return sum(self.config["visual_dims"]) + 18


def policy_contract(binding, space):
    if space not in ACTION_DIMS:
        raise ValueError("Unknown B1 action space")
    return dict(
        schema=CONTRACT_SCHEMA,
        action_schema=ACTION_SCHEMA,
        observation_schema=OBS_SCHEMA,
        arm_joints=list(ARM_JOINTS),
        observed_joints=list(ARM_JOINTS + NECK_JOINTS),
        obs_keys=list(OBS_KEYS),
        action_space=space,
        action_dim=ACTION_DIMS[space],
        obs_dim=binding.obs_dim,
        perception=binding.to_dict(),
    )


def validate_contract(value, binding=None, space=None):
    if not isinstance(value, dict) or "perception" not in value:
        raise ValueError("Missing explicit B1 policy contract; legacy data are not B1")
    recorded = PerceptionBinding.from_dict(value["perception"])
    expected = policy_contract(binding or recorded, space or value.get("action_space"))
    if value != expected:
        raise ValueError("B1 observation/action/perception contract mismatch")
    return recorded
