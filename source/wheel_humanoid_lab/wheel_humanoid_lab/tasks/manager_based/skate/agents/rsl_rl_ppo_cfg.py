from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class Q1SkatePPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """PPO for the Q1 skate task. Observation normalization is ON for actor and critic and is baked
    into the exported ONNX/JIT (play.py passes ``actor_obs_normalizer`` to the exporter)."""

    num_steps_per_env = 24
    max_iterations = 20000
    save_interval = 100
    experiment_name = "q1_skate_cubemars"
    obs_groups = {"policy": ["policy"], "critic": ["critic"]}
    clip_actions = 1.0
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
        # 0.01 let the per-dim std of the arm/torso/neck slots drift to ~2.5 (clip_actions=1.0 hides
        # the cost of a wide Gaussian), which turned the arms into bang-bang noise and starved the
        # arms_back / forward_lean terms. 0.003 keeps exploration without that runaway.
        entropy_coef=0.003,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
