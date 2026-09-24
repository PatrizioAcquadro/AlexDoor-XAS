"""Registration and configuration contracts for the Purdue environment."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from alexdoor_xas.envs import door_task

ENV_ID = "AlexDoor-DoorPush-Purdue-v0"
ENV_ENTRY_POINT = "alexdoor_xas.envs.door_task.door_push_purdue_env:DoorPushPurdueEnv"
CFG_ENTRY_POINT = "alexdoor_xas.envs.door_task.door_push_purdue_env_cfg:DoorPushPurdueEnvCfg"


def test_supported_environment_is_registered_if_gymnasium_available() -> None:
    if importlib.util.find_spec("gymnasium") is None:
        pytest.skip("gymnasium is not installed in this Python environment")

    import gymnasium as gym

    assert door_task.DOOR_PUSH_PURDUE_ENV_ID == ENV_ID
    spec = gym.spec(ENV_ID)
    assert spec.entry_point == ENV_ENTRY_POINT
    assert spec.kwargs["env_cfg_entry_point"] == CFG_ENTRY_POINT
    assert spec.disable_env_checker is True


def test_purdue_env_cfg_contract_if_isaaclab_available() -> None:
    if importlib.util.find_spec("isaaclab") is None:
        pytest.skip("isaaclab is not installed in this Python environment")

    from alexdoor_xas.envs.door_task.door_push_purdue_env_cfg import (
        DoorPushPurdueEnvCfg,
    )

    cfg = DoorPushPurdueEnvCfg()

    assert cfg.action_space == 6
    assert cfg.observation_space == 18
    assert cfg.state_space == 0
    assert cfg.sim.device == "cuda:0"
    assert cfg.sim.dt == pytest.approx(1 / 120)
    assert cfg.sim.render_interval == cfg.decimation
    assert cfg.scene.num_envs == 1
    assert cfg.cameras
    assert cfg.action_mode == "A2"


@pytest.mark.parametrize("failure", ["alex", "zed", "h5py"])
def test_preflight_reports_dependency_failures_without_crashing(monkeypatch, capsys, failure):
    import torch

    path = Path(__file__).resolve().parents[1] / "scripts" / "check_env.py"
    spec = importlib.util.spec_from_file_location("check_env", path)
    preflight = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(preflight)
    monkeypatch.setattr(preflight, "_check_provenance", lambda: ([], []))
    monkeypatch.setattr(preflight, "_pkg_version", lambda name: "installed")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda _: "preflight fixture")

    def unavailable(*args):
        raise ImportError(f"missing {failure}")

    if failure == "alex":
        monkeypatch.setattr(preflight.util, "find_spec", unavailable)
        monkeypatch.setattr(
            preflight, "_check_assets", lambda: pytest.fail("Alex import attempted")
        )
    else:
        monkeypatch.setattr(preflight, "_purdue_module_failure", lambda: None)
        monkeypatch.setattr(preflight, "_check_assets", unavailable if failure == "zed" else list)
    if failure == "h5py":
        monkeypatch.setattr(preflight, "PYTHON_PACKAGES", {"h5py": "h5py"})
        monkeypatch.setattr(preflight, "import_module", unavailable)

    assert preflight.main() == 1
    output = capsys.readouterr().out
    assert f"missing {failure}" in output
    assert output.rstrip().endswith("FAIL")
