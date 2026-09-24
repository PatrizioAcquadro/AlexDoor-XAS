"""Atomic checkpoints and random-state restoration for tensor training."""

import os
import random
from pathlib import Path
from typing import Any

import numpy as np


def torch_save_atomic(path: str | Path, payload: Any) -> Path:
    import torch

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    try:
        torch.save(payload, temporary)
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    return target


def capture_rng_states() -> dict[str, Any]:
    import torch

    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def restore_rng_states(states: dict[str, Any]) -> None:
    import torch

    random.setstate(states["python"])
    np.random.set_state(states["numpy"])
    torch.set_rng_state(states["torch_cpu"])
    if torch.cuda.is_available() and states.get("torch_cuda") is not None:
        torch.cuda.set_rng_state_all(states["torch_cuda"])
