import gymnasium as gym

from . import agents

gym.register(
    id="Isaac-WheelHumanoid-Skateboard-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.skateboard_env_cfg:WheelHumanoidSkateboardEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:WheelHumanoidSkateboardPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-WheelHumanoid-Skateboard-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.skateboard_env_cfg:WheelHumanoidSkateboardEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:WheelHumanoidSkateboardPPORunnerCfg",
    },
)
