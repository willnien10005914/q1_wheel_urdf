from isaaclab.utils import configclass
from .rsl_rl_ppo_cfg import Q1InertiaSkatePPORunnerCfg


@configclass
class NativePPORunnerCfg(Q1InertiaSkatePPORunnerCfg):
    experiment_name = "q1_native_manager"
    save_interval = 100
    max_iterations = 20000

    def __post_init__(self):
        self.policy.init_noise_std = .25
        self.algorithm.learning_rate = 3.e-4
        self.algorithm.entropy_coef = .002
