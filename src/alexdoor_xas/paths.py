"""Checkout-relative locations for local data."""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DATASETS_DIR = REPO_ROOT / "datasets"
