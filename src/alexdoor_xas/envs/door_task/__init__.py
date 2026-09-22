"""Register the single supported Purdue robot runtime."""

import gymnasium as gym

DOOR_PUSH_PURDUE_ENV_ID = "AlexDoor-DoorPush-Purdue-v0"

gym.register(
    id=DOOR_PUSH_PURDUE_ENV_ID,
    entry_point="alexdoor_xas.envs.door_task.door_push_purdue_env:DoorPushPurdueEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            "alexdoor_xas.envs.door_task.door_push_purdue_env_cfg:DoorPushPurdueEnvCfg"
        )
    },
)
