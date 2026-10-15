import gymnasium as gym

from . import agents

for _task_id, _env_cfg in (
    ("Isaac-Q1-Inertia-Skate-v0", "Q1InertiaSkateEnvCfg"),
    ("Isaac-Q1-Inertia-Skate-Play-v0", "Q1InertiaSkateEnvCfg_PLAY"),
):
    gym.register(
        id=_task_id,
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.inertia_env_cfg:{_env_cfg}",
            "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Q1InertiaSkatePPORunnerCfg",
        },
    )
