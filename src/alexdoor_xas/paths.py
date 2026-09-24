"""Canonical paths and identifiers for AlexDoor-XAS."""

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DATASETS_DIR = REPO_ROOT / "datasets"
OUTPUTS_DIR = REPO_ROOT / "outputs"
_RUNTIME_CACHE_ROOT = Path(
    os.environ.get("ALEXDOOR_CACHE_ROOT", str(Path.home() / ".cache" / "alexdoor-xas"))
).expanduser()
VERIFICATION_CACHE_DIR = _RUNTIME_CACHE_ROOT / "verification"
SCRIPTED_RUNS_CACHE_DIR = _RUNTIME_CACHE_ROOT / "scripted_runs"
