import gymnasium as gym

_TASKS = [
 ('', 'Q1RecoveryV3EnvCfg', 'Q1RecoveryV3PPORunnerCfg'),
 ('-Play', 'Q1RecoveryV3PlayCfg', 'Q1RecoveryV3PPORunnerCfg'),
 ('-Supine', 'Q1RecoveryV3SupineEnvCfg', 'Q1RecoveryV3SupinePPORunnerCfg'),
 ('-Supine-Play', 'Q1RecoveryV3SupinePlayCfg', 'Q1RecoveryV3SupinePPORunnerCfg'),
 ('-Prone', 'Q1RecoveryV3ProneEnvCfg', 'Q1RecoveryV3PronePPORunnerCfg'),
 ('-Prone-Play', 'Q1RecoveryV3PronePlayCfg', 'Q1RecoveryV3PronePPORunnerCfg'),
 ('-BootKneel', 'Q1RecoveryV3BootKneelEnvCfg', 'Q1RecoveryV3BootKneelPPORunnerCfg'),
 ('-BootKneel-Play', 'Q1RecoveryV3BootKneelPlayCfg', 'Q1RecoveryV3BootKneelPPORunnerCfg'),
]
for suffix, env_cfg, runner_cfg in _TASKS:
 gym.register(
  id=f'Isaac-Q1-RecoveryV3{suffix}-v0',
  entry_point='isaaclab.envs:ManagerBasedRLEnv',
  disable_env_checker=True,
  kwargs={
   'env_cfg_entry_point': f'{__name__}.env_cfg:{env_cfg}',
   'rsl_rl_cfg_entry_point': f'{__name__}.runner_cfg:{runner_cfg}',
  },
 )
