import gymnasium as gym

from . import agents

gym.register(
    id="Isaac-Q1-Slide-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.slide_env_cfg:Q1SlideEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Q1SlidePPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Q1-Slide-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.slide_env_cfg:Q1SlideEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Q1SlidePPORunnerCfg",
    },
)
