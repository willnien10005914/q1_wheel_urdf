import gymnasium as gym

from . import agents

gym.register(
    id="Isaac-Q1-Posture-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.posture_env_cfg:Q1PostureEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Q1PosturePPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Q1-Posture-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.posture_env_cfg:Q1PostureEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Q1PosturePPORunnerCfg",
    },
)
