from isaaclab.utils import configclass
from wheel_humanoid_lab.tasks.manager_based.recovery.runner_cfg import Q1RecoveryPPORunnerCfg

@configclass
class Q1RecoveryV3PPORunnerCfg(Q1RecoveryPPORunnerCfg):
 experiment_name='q1_recovery_contact_v3'
 # Mixed both-mode full runs plateaued ~10–12k; keep short default for smoke, override via CLI.
 max_iterations=300
 save_interval=50
 def __post_init__(self):
  # Keep exploration, but stop the previous full run's std ballooning to ~1.15 which
  # made standing arms thrash even after the motor prior settled.
  super().__post_init__()
  self.policy.init_noise_std=.05
  self.algorithm.entropy_coef=.0005
  self.algorithm.learning_rate=1.e-4
  self.algorithm.desired_kl=.01

@configclass
class Q1RecoveryV3SupinePPORunnerCfg(Q1RecoveryV3PPORunnerCfg):
 experiment_name='q1_recovery_v3_supine'
 max_iterations=12000
 save_interval=200

@configclass
class Q1RecoveryV3PronePPORunnerCfg(Q1RecoveryV3PPORunnerCfg):
 experiment_name='q1_recovery_v3_prone'
 max_iterations=12000
 save_interval=200
