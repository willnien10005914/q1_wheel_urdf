from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlPpoActorCriticCfg

from wheel_humanoid_lab.tasks.manager_based.skate.agents.rsl_rl_ppo_cfg import Q1SkatePPORunnerCfg


@configclass
class Q1PosturePPORunnerCfg(Q1SkatePPORunnerCfg):
    """Same network / normalizer layout as the skate PPO (hot-swappable ONNX), separate weights.

    The knee action scale is 1.6 rad, so start with a smaller exploration std than the skate run."""

    max_iterations = 4000
    experiment_name = "q1_posture_cubemars"
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.6,
        noise_std_type="scalar",
        actor_obs_normalization=True,
        critic_obs_normalization=True,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
