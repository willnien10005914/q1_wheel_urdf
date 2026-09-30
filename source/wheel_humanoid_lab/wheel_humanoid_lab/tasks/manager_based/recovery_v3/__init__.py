import gymnasium as gym
for suffix,cfg in [('', 'Q1RecoveryV3EnvCfg'),('-Play','Q1RecoveryV3PlayCfg')]:
 gym.register(id=f'Isaac-Q1-RecoveryV3{suffix}-v0',entry_point='isaaclab.envs:ManagerBasedRLEnv',disable_env_checker=True,kwargs={'env_cfg_entry_point':f'{__name__}.env_cfg:{cfg}','rsl_rl_cfg_entry_point':f'{__name__}.runner_cfg:Q1RecoveryV3PPORunnerCfg'})
