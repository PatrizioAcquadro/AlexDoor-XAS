"""Static, operational and offline evaluation entry points."""

import subprocess
from pathlib import Path


def source_commit() -> str | None:
    """Identify the checkout supplying diagnostics when Git metadata is available."""
    src = Path(__file__).resolve().parents[3]
    git_dir = src.parent / ".git"
    if src.name != "src" or not git_dir.exists():
        return None
    try:
        return subprocess.check_output(
            ["git", "--git-dir", str(git_dir), "rev-parse", "--verify", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None
