"""Explicit B1 compatibility and the still-pending Phase 6.0 release boundary."""

import json
from dataclasses import dataclass

from alexdoor_xas.action.b1 import ACTION_DIMS, ACTION_SCHEMA
from alexdoor_xas.assets.purdue import ARM_JOINTS, NECK_JOINTS

OBS_KEYS = ("rgbd_static", "rgbd_recent", "joint_position", "joint_velocity")
OBS_SCHEMA = "b1.policy-observation.v1"
CONTRACT_SCHEMA = "b1.policy-contract.v1"
RELEASE_SCHEMA = "b1.perception.release.v1"


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
        for key in ("checkpoint_sha256", "backbone_sha256"):
            check_sha(release.get(key))
        cfg = release.get("config", {})
        if (
            cfg.get("model") != "metric-memory-v1"
            or not cfg.get("inspection")
            or not isinstance(cfg.get("hidden_size"), int)
            or cfg["hidden_size"] < 1
            or not 0 < cfg.get("max_gap_s", 0)
            or not 0 < cfg.get("sample_hz", 0)
        ):
            raise ValueError("Incompatible B1 perception recipe")

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
        return 2 * self.config["hidden_size"] + 18


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
