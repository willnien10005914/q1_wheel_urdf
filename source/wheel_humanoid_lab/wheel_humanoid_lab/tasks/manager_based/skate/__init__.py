import gymnasium as gym

from . import agents

gym.register(
    id="Isaac-Q1-Skate-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.skate_env_cfg:Q1SkateEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Q1SkatePPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Q1-Skate-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.skate_env_cfg:Q1SkateEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Q1SkatePPORunnerCfg",
    },
)
