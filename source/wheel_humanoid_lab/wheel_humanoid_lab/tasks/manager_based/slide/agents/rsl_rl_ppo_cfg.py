from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg

from wheel_humanoid_lab.tasks.manager_based.skate.agents.rsl_rl_ppo_cfg import Q1SkatePPORunnerCfg


@configclass
class Q1SlidePPORunnerCfg(Q1SkatePPORunnerCfg):
    """Same net as skate. Noise is reopened after the warm start (see RESET_NOISE_STD) so the
    stride can be explored; entropy stays high enough that std does not collapse back to a plant."""

    max_iterations = 12000
    experiment_name = "q1_slide_cubemars"
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
        entropy_coef=0.008,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
