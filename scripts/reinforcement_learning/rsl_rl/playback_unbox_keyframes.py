#!/usr/bin/env python3
"""Headless Isaac Lab playback of web/MediaPipe unbox keyframes → RGB mp4.

Drives the Q1 articulation through the recorded knots (same JSON the web UI saves),
holding each pose with write_joint_state so you can check the *designed* motion
independent of the trained PPO.

Usage (via ./run_isaac.sh or ./record_unbox_keyframes.sh):
  python scripts/reinforcement_learning/rsl_rl/playback_unbox_keyframes.py \\
      --keyframes docs/reference/a3_unbox_ref_web_recording.json \\
      --headless --video --enable_cameras
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Playback unbox keyframes and record RGB video.")
parser.add_argument("--keyframes", type=str, required=True, help="web recording / keyframes JSON")
parser.add_argument("--video", action="store_true", default=True)
parser.add_argument("--video_length", type=int, default=0, help="0 = auto from duration")
parser.add_argument("--hold_s", type=float, default=0.55, help="seconds held on each knot")
parser.add_argument("--blend_s", type=float, default=0.70, help="seconds blending between knots")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--task", type=str, default="Isaac-Q1-Unbox-Play-v0")
parser.add_argument("--out", type=str, default="", help="copy final mp4 here")
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
args_cli.enable_cameras = True
args_cli.headless = True
sys.argv = [sys.argv[0]] + hydra_args
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab_tasks.utils.hydra import hydra_task_config

import wheel_humanoid_lab.tasks  # noqa: F401


def _load_knots(path: str) -> list[dict]:
    doc = json.load(open(path))
    kfs = doc.get("keyframes") or []
    if not kfs:
        raise SystemExit(f"no keyframes in {path}")
    return kfs


def _pose_to_q(asset, pose_regex: dict | None, joints: dict | None) -> torch.Tensor:
    q = asset.data.default_joint_pos[0].clone()
    names = asset.joint_names
    if joints:
        for i, n in enumerate(names):
            if n in joints:
                q[i] = float(joints[n])
    if pose_regex:
        for i, n in enumerate(names):
            for pat, val in pose_regex.items():
                if re.fullmatch(pat, n):
                    q[i] = float(val)
    return q


def _lerp(a: torch.Tensor, b: torch.Tensor, u: float) -> torch.Tensor:
    u = max(0.0, min(1.0, u))
    s = u * u * (3.0 - 2.0 * u)
    return a * (1.0 - s) + b * s


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg, agent_cfg):
    knots = _load_knots(args_cli.keyframes)
    n = len(knots)
    dt_hold = args_cli.hold_s
    dt_blend = args_cli.blend_s
    total_s = n * dt_hold + max(0, n - 1) * dt_blend + 0.4
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = args_cli.seed
    env_cfg.sim.device = args_cli.device if hasattr(args_cli, "device") else "cuda:0"

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    out_dir = os.path.join(project_root, "checkpoints", "videos")
    os.makedirs(out_dir, exist_ok=True)
    log_dir = os.path.join(out_dir, "unbox_keyframes_play")
    os.makedirs(log_dir, exist_ok=True)
    env_cfg.log_dir = log_dir

    step_dt = float(env_cfg.sim.dt) * env_cfg.decimation
    n_steps = args_cli.video_length or max(30, int(math.ceil(total_s / step_dt)))

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array")
    env = gym.wrappers.RecordVideo(
        env,
        video_folder=os.path.join(log_dir, "videos", "play"),
        step_trigger=lambda step: step == 0,
        video_length=n_steps,
        disable_logger=True,
    )
    print(f"[INFO] Playing {n} keyframes over ~{total_s:.1f}s ({n_steps} steps @ {1.0/step_dt:.0f} Hz)")

    obs, _ = env.reset()
    robot = env.unwrapped.scene["robot"]
    device = env.unwrapped.device
    env_ids = torch.tensor([0], device=device, dtype=torch.long)

    qs = []
    zs = []
    for k in knots:
        qs.append(_pose_to_q(robot, k.get("pose_regex"), k.get("joints")).to(device))
        zs.append(float(k.get("pelvis_z", 0.30)))

    # Start supine-ish at first knot.
    q0 = qs[0].unsqueeze(0)
    robot.write_joint_state_to_sim(q0, torch.zeros_like(q0), env_ids=env_ids)
    root = robot.data.root_state_w[env_ids].clone()
    root[:, 2] = zs[0] + env.unwrapped.scene.env_origins[env_ids, 2]
    # Face up (same as reset_supine).
    root[:, 3:7] = torch.tensor([0.70710678, 0.0, 0.70710678, 0.0], device=device)
    root[:, 7:] = 0.0
    robot.write_root_pose_to_sim(root[:, :7], env_ids=env_ids)
    robot.write_root_velocity_to_sim(root[:, 7:], env_ids=env_ids)

    def set_pose(q: torch.Tensor, z: float, upright_blend: float):
        """upright_blend 0=supine quat, 1=standing/kneel upright quat."""
        qq = q.unsqueeze(0)
        robot.write_joint_state_to_sim(qq, torch.zeros_like(qq), env_ids=env_ids)
        root = robot.data.root_state_w[env_ids].clone()
        root[:, 2] = z + env.unwrapped.scene.env_origins[env_ids, 2]
        # slerp-ish between face-up and identity
        u = max(0.0, min(1.0, upright_blend))
        # face-up: wxyz ≈ (0.707, 0, 0.707, 0); upright: (1,0,0,0)
        w = 0.70710678 * (1.0 - u) + 1.0 * u
        y = 0.70710678 * (1.0 - u)
        nrm = math.sqrt(w * w + y * y) + 1e-9
        root[:, 3:7] = torch.tensor([w / nrm, 0.0, y / nrm, 0.0], device=device)
        root[:, 7:] = 0.0
        robot.write_root_pose_to_sim(root[:, :7], env_ids=env_ids)
        robot.write_root_velocity_to_sim(root[:, 7:], env_ids=env_ids)

    # Dummy zero action (we overwrite state each step).
    action_dim = env.action_space.shape[-1]
    zero = torch.zeros((1, action_dim), device=device)

    schedule = []  # list of (q, z, upright)
    for i in range(n):
        upright = i / max(n - 1, 1)
        hold_steps = max(1, int(round(dt_hold / step_dt)))
        for _ in range(hold_steps):
            schedule.append((qs[i], zs[i], upright))
        if i + 1 < n:
            blend_steps = max(1, int(round(dt_blend / step_dt)))
            for s in range(blend_steps):
                u = (s + 1) / blend_steps
                schedule.append((_lerp(qs[i], qs[i + 1], u), zs[i] * (1 - u) + zs[i + 1] * u, (i + u) / max(n - 1, 1)))

    while len(schedule) < n_steps:
        schedule.append(schedule[-1])
    schedule = schedule[:n_steps]

    for step, (q, z, up) in enumerate(schedule):
        set_pose(q, z, up)
        env.step(zero)
        if (step + 1) % 50 == 0:
            print(f"[INFO] step {step+1}/{n_steps}", flush=True)

    env.close()
    src = os.path.join(log_dir, "videos", "play", "rl-video-step-0.mp4")
    out = args_cli.out or os.path.join(out_dir, "q1_unbox_keyframes.mp4")
    if os.path.isfile(src):
        import shutil

        shutil.copy2(src, out)
        print(f"[INFO] Video saved to: {out}")
    else:
        print(f"[WARN] Expected video missing: {src}")
    simulation_app.close()


if __name__ == "__main__":
    main()
