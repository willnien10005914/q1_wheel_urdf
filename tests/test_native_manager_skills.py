"""Bounded native manager contract and zero-residual physical smoke test."""
import argparse
import json
from pathlib import Path
from isaaclab.app import AppLauncher
p = argparse.ArgumentParser()
p.add_argument("--skill", default="Stand", choices=["Stand", "Walk", "Standup", "Backflip", "Web"])
p.add_argument("--steps", type=int, default=200)
p.add_argument("--out", type=Path, required=True)
AppLauncher.add_app_launcher_args(p)
a = p.parse_args(); a.headless = True; app = AppLauncher(a).app
import gymnasium as gym
import torch
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab_tasks.utils import parse_env_cfg
import wheel_humanoid_lab.tasks

task = f"Isaac-Q1-Native-{a.skill}-v0"
env = None
try:
    cfg = parse_env_cfg(task, device=a.device or "cuda:0", num_envs=2)
    cfg.seed = 42
    env = gym.make(task, cfg=cfg).unwrapped
    assert isinstance(env, ManagerBasedRLEnv)
    obs, _ = env.reset(seed=42)
    robot = env.scene["robot"]
    assert env.action_manager.total_action_dim == 23
    assert obs["policy"].shape == (2, 90) and obs["critic"].shape == (2, 90)
    assert robot.num_joints == 27 and not any("neck" in name for name in robot.joint_names)
    assert "q1_inertia" in cfg.scene.robot.spawn.asset_path
    s = env.command_manager.get_term("skill")
    before_root = robot.data.root_state_w.clone(); before_q = robot.data.joint_pos.clone()
    s.set_mode(a.skill.lower() if a.skill != "Web" else "stand")
    assert torch.equal(before_root, robot.data.root_state_w) and torch.equal(before_q, robot.data.joint_pos)
    if a.skill == "Backflip": s.level = 1.
    done_count = 0; min_z = 10.; upright_steps = 0; nonfinite = 0; max_turn = 0.
    for step in range(a.steps):
        obs, reward, term, trunc, _ = env.step(torch.zeros(2, 23, device=env.device))
        assert torch.isfinite(obs["policy"]).all() and torch.isfinite(reward).all()
        nonfinite += int(env.termination_manager.get_term("nonfinite").sum())
        done_count += int(term.sum()); z = robot.data.root_pos_w[:, 2]-env.scene.env_origins[:, 2]
        min_z = min(min_z, float(z.min()))
        upright_steps += int(((-robot.data.projected_gravity_b[:, 2] > .9) & (z > .65)).sum())
        max_turn = max(max_turn, float((-s.winding).max()))
    assert nonfinite == 0
    result = dict(task=task,manager_based=True,new_urdf=True,action_dim=23,actor_obs_dim=90,critic_obs_dim=90,
                  steps=a.steps,num_envs=2,terminations=done_count,min_height_m=min_z,
                  upright_fraction=upright_steps/(2*a.steps),max_backturn_rad=max_turn,
                  handover_no_pose_writes=True,nonfinite=nonfinite)
    a.out.parent.mkdir(parents=True, exist_ok=True); a.out.write_text(json.dumps(result,indent=2))
    print("NATIVE_MANAGER_CONTRACT", json.dumps(result), flush=True)
finally:
    if env is not None: env.close()
    app.close()
