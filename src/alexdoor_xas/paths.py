"""Canonical paths and identifiers for AlexDoor-XAS."""

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DATASETS_DIR = REPO_ROOT / "datasets"
OUTPUTS_DIR = REPO_ROOT / "outputs"
PHASE4_1_ASSETS_DIR = REPO_ROOT / "assets" / "doors" / "phase4_1"
PHASE4_1_SOURCE_DIR = PHASE4_1_ASSETS_DIR / "source"
PHASE4_1_NORMALIZED_DIR = PHASE4_1_ASSETS_DIR / "normalized"
PHASE4_1_EVIDENCE_DIR = PHASE4_1_ASSETS_DIR / "evidence"
PHASE4_1_RECIPES_DIR = PHASE4_1_ASSETS_DIR / "recipes"
PHASE4_1_MANIFEST = PHASE4_1_ASSETS_DIR / "manifest.json"
ALEX_V2_CALIBRATION = REPO_ROOT / "configs" / "alex_v2_door.json"

_RUNTIME_CACHE_ROOT = Path(
    os.environ.get("ALEXDOOR_CACHE_ROOT", str(Path.home() / ".cache" / "alexdoor-xas"))
).expanduser()
VERIFICATION_CACHE_DIR = _RUNTIME_CACHE_ROOT / "verification"
SCRIPTED_RUNS_CACHE_DIR = _RUNTIME_CACHE_ROOT / "scripted_runs"

ALEX_V2_ASSET_ROOT = (
    Path(os.environ.get("ALEX_V2_ASSET_ROOT", str(Path.home() / "Desktop" / "Alex")))
    .expanduser()
    .resolve()
)
ALEX_V2_URDF = ALEX_V2_ASSET_ROOT / "urdf" / "alex_v2.urdf"

ALEX_V2_TASK = "door_push_alex_v2"
ALEX_V2_DATASET_VERSION = "v2_pose"
ALEX_V2_ROBOT_TAG = "alex_v2_fullbody_fixedbase_standard_forearm_v0"
DOOR_SCENE_DIR = OUTPUTS_DIR / "door_scene"
DOOR_USD = (
    Path(os.environ.get("ALEXDOOR_ASSETS_ROOT", str(Path.home() / "Desktop"))).expanduser()
    / "CombinedScene"
    / "Door.usd"
)
