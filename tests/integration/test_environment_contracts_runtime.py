"""Runtime-specific checks separated from portable behavioral tests."""

from __future__ import annotations

import importlib.util

import pytest


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
