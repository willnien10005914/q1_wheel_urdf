import gymnasium as gym
for _name in ('Standup','Kneel','Walk','Backflip','Skate','Web'):
    gym.register(id=f'Isaac-Q1-Reference-{_name}-v0',entry_point='isaaclab.envs:ManagerBasedRLEnv',disable_env_checker=True,
        kwargs={'env_cfg_entry_point':f'{__name__}.env_cfg:Reference{_name}EnvCfg',
                'rsl_rl_cfg_entry_point':'wheel_humanoid_lab.tasks.manager_based.inertia.agents.native_ppo_cfg:NativePPORunnerCfg'})
