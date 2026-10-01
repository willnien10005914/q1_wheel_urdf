from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg

from wheel_humanoid_lab.tasks.manager_based.skate.agents.rsl_rl_ppo_cfg import Q1SkatePPORunnerCfg


@configclass
class Q1UnboxPPORunnerCfg(Q1SkatePPORunnerCfg):
    """Same 92/24 net as the other skills. Fresh std: do not resume skate/slide/getup."""

    max_iterations = 8000
    experiment_name = "q1_unbox_cubemars"
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        noise_std_type="scalar",
        actor_obs_normalization=True,
        critic_obs_normalization=True,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        # The first get-up run used 0.008 and collapsed to std 0.03 while still supine.
        entropy_coef=0.012,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=5.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.016,
        max_grad_norm=1.0,
    )
