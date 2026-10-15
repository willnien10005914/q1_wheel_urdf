"""View a native 23-action Q1 inertia checkpoint in its own manager-based task."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time

from isaaclab.app import AppLauncher

ROOT = Path(__file__).resolve().parents[3]
TASK = "Isaac-Q1-Inertia-Skate-Play-v0"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint", type=Path,
        default=ROOT / "checkpoints/inertia_native_smoke/model_19.pt",
        help="A checkpoint trained for the native 23-action inertia task.",
    )
    parser.add_argument("--num_envs", type=int, default=1)
    parser.add_argument("--steps", type=int, default=0, help="Stop after this many policy steps; 0 plays until closed.")
    parser.add_argument("--seed", type=int, default=4107)
    parser.add_argument("--vx", type=float, default=0.0, help="Fixed forward velocity command in m/s.")
    parser.add_argument("--yaw", type=float, default=0.0, help="Fixed yaw rate command in rad/s.")
    parser.add_argument("--real-time", action="store_true", help="Pace policy steps to the simulation control rate.")
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    if args.num_envs < 1 or args.steps < 0 or args.seed < 0:
        parser.error("--num_envs must be positive; --steps and --seed must be nonnegative")
    if not all(math.isfinite(value) for value in (args.vx, args.yaw)):
        parser.error("--vx and --yaw must be finite")
    checkpoint = args.checkpoint.expanduser().resolve()
    if not checkpoint.is_file():
        parser.error(f"Native inertia checkpoint does not exist: {checkpoint}")

    app = AppLauncher(args).app
    env = None
    try:
        import gymnasium as gym
        import torch
        from rsl_rl.runners import OnPolicyRunner
        from isaaclab.envs import ManagerBasedRLEnv, ManagerBasedRLEnvCfg
        from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
        from isaaclab_tasks.utils import load_cfg_from_registry, parse_env_cfg
        import wheel_humanoid_lab.tasks  # noqa: F401
        from runtime_compat import load_checkpoint, policy_module, policy_observations, runner_cfg

        cfg = parse_env_cfg(TASK, device=args.device or "cuda:0", num_envs=args.num_envs)
        if not isinstance(cfg, ManagerBasedRLEnvCfg):
            raise TypeError(f"{TASK} requires ManagerBasedRLEnvCfg")
        cfg.seed = args.seed
        agent_cfg = load_cfg_from_registry(TASK, "rsl_rl_cfg_entry_point")
        agent_cfg.seed = args.seed
        agent_cfg.device = cfg.sim.device
        env = gym.make(TASK, cfg=cfg)
        if not isinstance(env.unwrapped, ManagerBasedRLEnv):
            raise TypeError(f"{TASK} must create a ManagerBasedRLEnv")
        env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
        runner = OnPolicyRunner(env, runner_cfg(agent_cfg), log_dir=None, device=cfg.sim.device)
        load_checkpoint(runner, checkpoint, load_optimizer=False)
        policy = runner.get_inference_policy(device=cfg.sim.device)
        model = policy_module(runner)
        base = env.unwrapped
        robot = base.scene["robot"]
        term = base.command_manager.get_term("base_velocity")
        command = base.command_manager.get_command("base_velocity")
        target = torch.tensor([args.vx, 0.0, args.yaw], device=command.device, dtype=command.dtype)

        def hold_command():
            command[:] = target
            term.time_left.fill_(1.0e9)
            term.is_standing_env.fill_(args.vx == 0.0 and args.yaw == 0.0)
            if hasattr(term, "is_heading_env"):
                term.is_heading_env.fill_(False)

        def observations():
            value = env.get_observations()
            return value[0] if isinstance(value, tuple) else value

        env.seed(args.seed)
        env.reset()
        hold_command()
        obs = observations()
        policy_obs = obs if torch.is_tensor(obs) else obs["policy"]
        print("INERTIA_PLAY", json.dumps({
            "task": TASK, "manager_based": True, "checkpoint": str(checkpoint),
            "num_envs": base.num_envs, "observation_dim": policy_obs.shape[-1],
            "action_dim": base.action_manager.total_action_dim,
            "command": [args.vx, 0.0, args.yaw], "step_dt": base.step_dt,
            "requested_steps": args.steps,
        }), flush=True)

        steps = done_count = 0
        try:
            with torch.inference_mode():
                while app.is_running() and (args.steps == 0 or steps < args.steps):
                    started = time.perf_counter()
                    # Override the command after automatic resets, then recompute
                    # observations so the actor receives exactly this command.
                    hold_command()
                    obs = observations()
                    actions = policy(policy_observations(obs))
                    if not bool(torch.isfinite(actions).all()):
                        raise RuntimeError("Native inertia policy produced non-finite actions")
                    _, _, done, _ = env.step(actions)
                    if not bool(torch.isfinite(robot.data.root_state_w).all()):
                        raise RuntimeError("Native inertia robot produced a non-finite root state")
                    done_count += int(done.sum().item())
                    if hasattr(model, "reset"):
                        model.reset(done)
                    steps += 1
                    if args.real_time:
                        time.sleep(max(0.0, base.step_dt - (time.perf_counter() - started)))
        except KeyboardInterrupt:
            if args.steps:
                raise
        if args.steps and steps != args.steps:
            raise RuntimeError(f"Simulator closed after {steps}/{args.steps} requested steps")
        print("INERTIA_PLAY_RESULT", json.dumps({
            "steps": steps, "done_count": done_count,
            "root_height_m": (robot.data.root_pos_w[:, 2] - base.scene.env_origins[:, 2]).tolist(),
        }, allow_nan=False), flush=True)
    finally:
        try:
            if env is not None:
                env.close()
        finally:
            app.close()


if __name__ == "__main__":
    main()
