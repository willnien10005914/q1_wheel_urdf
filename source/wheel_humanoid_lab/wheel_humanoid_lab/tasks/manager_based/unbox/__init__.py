import gymnasium as gym

from . import agents

gym.register(
    id="Isaac-Q1-Unbox-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.unbox_env_cfg:Q1UnboxEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Q1UnboxPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Q1-Unbox-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.unbox_env_cfg:Q1UnboxEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Q1UnboxPPORunnerCfg",
    },
)
