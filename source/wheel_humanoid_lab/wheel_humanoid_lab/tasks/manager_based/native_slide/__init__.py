import gymnasium as gym

for name, cfg in [('Slide', 'NativeSlideEnvCfg'), ('Slide-Web', 'NativeSlideWebEnvCfg')]:
    gym.register(id=f'Isaac-Q1-Native-{name}-v0', entry_point='isaaclab.envs:ManagerBasedRLEnv',
                 disable_env_checker=True,
                 kwargs={'env_cfg_entry_point':f'{__name__}.env_cfg:{cfg}',
                         'rsl_rl_cfg_entry_point':'wheel_humanoid_lab.tasks.manager_based.inertia.agents.native_ppo_cfg:NativePPORunnerCfg'})
