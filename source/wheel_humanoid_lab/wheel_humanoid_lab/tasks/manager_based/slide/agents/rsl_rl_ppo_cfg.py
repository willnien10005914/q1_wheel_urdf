from isaaclab.utils import configclass

from wheel_humanoid_lab.tasks.manager_based.skate.agents.rsl_rl_ppo_cfg import Q1SkatePPORunnerCfg


@configclass
class Q1SlidePPORunnerCfg(Q1SkatePPORunnerCfg):
    """Identical network to the skate PPO: ``train_slide.sh`` resumes from ``checkpoints/q1_skate_ppo.pt``
    and fine-tunes with the stride rewards, so the balance skill is kept."""

    max_iterations = 3000
    experiment_name = "q1_slide_cubemars"
