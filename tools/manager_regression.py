#!/usr/bin/env python3
"""Bounded manager-based policy rollout capture and CPU-only behavior comparison.

Capture never calls learn/save and never publishes a checkpoint. Observations and
actions are recorded before each step; rewards/dones and robot state are recorded
after the step (Isaac Lab automatically resets terminated environments).
"""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]


def _flatten(value):
    if isinstance(value, dict):
        for key in sorted(value):
            yield from _flatten(value[key])
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _flatten(item)
    else:
        yield float(value)


def _metric(left, right):
    a, b = list(_flatten(left)), list(_flatten(right))
    if len(a) != len(b) or not a:
        raise ValueError(f"Mismatched or empty metric sizes: {len(a)} / {len(b)}")
    if not all(math.isfinite(value) for value in a + b):
        raise ValueError("Non-finite rollout value")
    errors = [abs(x - y) for x, y in zip(a, b)]
    rmse = math.sqrt(sum(error * error for error in errors) / len(errors))
    scale = max(math.sqrt(sum(value * value for value in a) / len(a)), 0.01)
    return {"rmse": rmse, "max_abs": max(errors), "relative_rmse": rmse / scale}


def compare_rollouts(baseline, candidate, thresholds, allow_checkpoint_change=False):
    failures, metrics = [], {}
    left, right = baseline["metadata"], candidate["metadata"]
    for key in ("task", "seed", "steps", "num_envs", "step_dt", "schema", "fixed_commands"):
        if left.get(key) != right.get(key):
            failures.append(f"Metadata differs: {key}")
    if not allow_checkpoint_change and left["checkpoint_sha256"] != right["checkpoint_sha256"]:
        failures.append("Checkpoint hashes differ; use --allow-checkpoint-change only for an intentional PPO update")
    a, b = baseline["trace"], candidate["trace"]
    if len(a) != len(b) or len(a) != left["steps"] or len(b) != right["steps"]:
        failures.append("Rollout lengths differ or do not match the requested finite step count")
    if not failures:
        for key in ("observation", "action", "reward", "height", "position", "gravity", "command"):
            try:
                metrics[key] = _metric([row[key] for row in a], [row[key] for row in b])
                metric_name = "relative_rmse" if key == "reward" else "rmse"
                if metrics[key][metric_name] > thresholds[key]:
                    failures.append(f"{key} {metric_name} {metrics[key][metric_name]:.6g} > {thresholds[key]:.6g}")
            except (KeyError, ValueError, TypeError) as error:
                failures.append(f"{key}: {error}")
        for key in ("done", "terminated", "truncated"):
            try:
                av = list(_flatten([row[key] for row in a]))
                bv = list(_flatten([row[key] for row in b]))
                if len(av) != len(bv) or not all(value in (0.0, 1.0) for value in av + bv):
                    raise ValueError("Invalid termination arrays")
                delta = abs(sum(av) - sum(bv))
                metrics[key] = {"baseline_count": sum(av), "candidate_count": sum(bv),
                                "count_delta": delta,
                                "step_mismatches": sum(x != y for x, y in zip(av, bv))}
                if delta > thresholds["termination_count"]:
                    failures.append(f"{key} count delta {delta} > {thresholds['termination_count']}")
            except (KeyError, ValueError, TypeError) as error:
                failures.append(f"{key}: {error}")
    return {"pass": not failures, "failures": failures, "thresholds": thresholds,
            "checkpoint_change_allowed": allow_checkpoint_change, "metrics": metrics}


def _write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    # Strict serialization also catches non-finite values in fields outside the
    # primary comparison metrics. Replace only this report, never model files.
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def capture(argv):
    # Resolve source/assets before importing any Isaac Lab or Q1 task module.
    bootstrap = argparse.ArgumentParser(add_help=False)
    bootstrap.add_argument("--project-root", type=Path, default=ROOT)
    selected, _ = bootstrap.parse_known_args(argv)
    project = selected.project_root.expanduser().resolve()
    os.environ["WHEEL_HUMANOID_ROOT"] = str(project)
    sys.path[:0] = [str(project / "source/wheel_humanoid_lab"), str(project),
                    str(ROOT / "scripts/reinforcement_learning/rsl_rl")]
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description="Capture a finite manager-based deterministic policy rollout.")
    parser.add_argument("--project-root", type=Path, default=project)
    parser.add_argument("--task", default="Isaac-Q1-Skate-Play-v0")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=128)
    parser.add_argument("--num_envs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=4107)
    parser.add_argument("--command", action="append", default=[], metavar="TERM=V1,V2,...",
                        help="Hold a named manager command fixed, e.g. base_velocity=0,0,0 or posture=0")
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args(argv)
    if not 1 <= args.steps <= 10000 or not 1 <= args.num_envs <= 16:
        parser.error("Bounded validation requires 1..10000 steps and 1..16 environments")
    if args.seed < 0:
        parser.error("A nonnegative fixed seed is required")
    checkpoint = args.checkpoint.expanduser().resolve(strict=True)
    fixed_commands = {}
    for raw in args.command:
        try:
            name, values = raw.split("=", 1)
            values = [float(value) for value in values.split(",")]
            if not name or not values or not all(math.isfinite(value) for value in values):
                raise ValueError()
            fixed_commands[name] = values
        except ValueError:
            parser.error(f"Invalid --command {raw!r}; expected TERM=V1,V2,... with finite values")
    args.headless = True
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
        from runtime_compat import load_checkpoint, runner_cfg

        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        cfg = parse_env_cfg(args.task, device=args.device or "cuda:0", num_envs=args.num_envs)
        if not isinstance(cfg, ManagerBasedRLEnvCfg):
            raise TypeError(f"{args.task} does not use ManagerBasedRLEnvCfg")
        cfg.seed = args.seed
        agent_cfg = load_cfg_from_registry(args.task, "rsl_rl_cfg_entry_point")
        agent_cfg.seed = args.seed
        agent_cfg.device = cfg.sim.device
        env = gym.make(args.task, cfg=cfg)
        if not isinstance(env.unwrapped, ManagerBasedRLEnv):
            raise TypeError(f"{args.task} did not create a ManagerBasedRLEnv")
        env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
        runner = OnPolicyRunner(env, runner_cfg(agent_cfg), log_dir=None, device=cfg.sim.device)
        load_checkpoint(runner, str(checkpoint), load_optimizer=False)
        policy = runner.get_inference_policy(device=cfg.sim.device)
        base = env.unwrapped
        robot = base.scene["robot"]

        def hold_commands():
            for name, values in fixed_commands.items():
                term = base.command_manager.get_term(name)
                command = base.command_manager.get_command(name)
                if len(values) != command.shape[-1]:
                    raise ValueError(f"Command {name} needs {command.shape[-1]} values, got {len(values)}")
                command[:] = torch.tensor(values, device=command.device)
                term.time_left.fill_(1.0e9)
                if hasattr(term, "is_standing_env"):
                    term.is_standing_env.fill_(not any(values))
                if hasattr(term, "is_heading_env"):
                    term.is_heading_env.fill_(False)

        def observations():
            value = env.get_observations()
            return value[0] if isinstance(value, tuple) else value

        def tensor_list(value):
            if not bool(torch.isfinite(value).all()):
                raise ValueError("Rollout produced a non-finite tensor")
            return value.detach().cpu().tolist()

        def policy_obs(value):
            return value["policy"] if not isinstance(value, torch.Tensor) else value

        # Runner construction may initialize observation statistics. Reseed and
        # reset afterwards so random initialization cannot shift trial inputs.
        env.seed(args.seed)
        env.reset()
        hold_commands()
        obs = observations()
        metadata = {"format_version": 1, "task": args.task, "manager_based": True,
                    "project_root": str(project), "seed": args.seed, "steps": args.steps,
                    "num_envs": args.num_envs, "step_dt": float(base.step_dt),
                    "checkpoint": str(checkpoint),
                    "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                    "fixed_commands": fixed_commands,
                    "schema": {"observation_shape": list(policy_obs(obs).shape),
                               "action_dim": base.action_manager.total_action_dim,
                               "joint_names": list(robot.joint_names),
                               "observation_terms": base.observation_manager.active_terms,
                               "reward_terms": list(base.reward_manager.active_terms),
                               "command_shapes": {name: list(base.command_manager.get_command(name).shape)
                                                  for name in base.command_manager.active_terms}},
                    "state_timing": "post-step, including automatic resets; observation/action are pre-step"}
        trace = []
        with torch.inference_mode():
            for step in range(args.steps):
                if not app.is_running():
                    raise RuntimeError(f"Simulator exited before the requested step count ({step}/{args.steps})")
                if fixed_commands:
                    hold_commands()
                    obs = observations()
                action = policy(obs)
                row = {"step": step, "observation": tensor_list(policy_obs(obs)),
                       "action": tensor_list(action),
                       "command": {name: tensor_list(base.command_manager.get_command(name))
                                   for name in base.command_manager.active_terms}}
                obs, reward, done, extras = env.step(action)
                if isinstance(obs, tuple):
                    obs = obs[0]
                position = robot.data.root_pos_w - base.scene.env_origins
                row.update({"reward": tensor_list(reward), "done": tensor_list(done),
                            "terminated": tensor_list(base.reset_terminated),
                            "truncated": tensor_list(base.reset_time_outs),
                            "height": tensor_list(position[:, 2]), "position": tensor_list(position),
                            "gravity": tensor_list(robot.data.projected_gravity_b)})
                trace.append(row)
                module = getattr(policy, "__self__", None)
                if hasattr(module, "reset"):
                    module.reset(done)
        result = {"metadata": metadata, "trace": trace,
                  "summary": {"steps": len(trace),
                              "mean_reward": sum(_flatten([row["reward"] for row in trace])) / (args.steps * args.num_envs),
                              "min_height": min(_flatten([row["height"] for row in trace])),
                              "done_count": sum(_flatten([row["done"] for row in trace]))}}
        _write_json(args.out, result)
        print("MANAGER_CAPTURE", json.dumps({"out": str(args.out), **result["summary"]}), flush=True)
    finally:
        if env is not None:
            env.close()
        app.close()


def compare(argv):
    parser = argparse.ArgumentParser(description="Compare recorded manager rollouts without loading Isaac Sim or Torch.")
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--allow-checkpoint-change", action="store_true")
    defaults = {"observation": 0.1, "action": 0.05, "reward": 0.1, "height": 0.02,
                "position": 0.1, "gravity": 0.05, "command": 1e-6, "termination_count": 0}
    for key, default in defaults.items():
        parser.add_argument("--max-" + key.replace("_", "-"), type=float, default=default,
                            help=f"Maximum {'relative RMSE' if key == 'reward' else 'count delta' if key == 'termination_count' else 'RMSE'} (default {default})")
    args = parser.parse_args(argv)
    thresholds = {key: getattr(args, "max_" + key) for key in defaults}
    if not all(math.isfinite(value) and value >= 0 for value in thresholds.values()):
        parser.error("Thresholds must be finite and nonnegative")
    report = compare_rollouts(json.loads(args.baseline.read_text()), json.loads(args.candidate.read_text()),
                              thresholds, args.allow_checkpoint_change)
    if args.out:
        _write_json(args.out, report)
    print(json.dumps(report, indent=2, allow_nan=False))
    return 0 if report["pass"] else 1


def policy_compare(argv):
    """Compare policies on identical recorded observations, avoiding rollout drift."""
    parser = argparse.ArgumentParser(description="CPU-only checkpoint comparison on identical reference observations.")
    parser.add_argument("reference", type=Path, help="JSON produced by the capture command")
    parser.add_argument("--baseline-checkpoint", type=Path, required=True)
    parser.add_argument("--candidate-checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--max-action-rmse", type=float, default=0.05)
    args = parser.parse_args(argv)
    if not math.isfinite(args.max_action_rmse) or args.max_action_rmse < 0:
        parser.error("--max-action-rmse must be finite and nonnegative")
    sys.path.insert(0, str(ROOT))
    import torch
    from skills.policy import FrozenActor

    reference = json.loads(args.reference.read_text())
    observations = [observation for row in reference["trace"] for observation in row["observation"]]
    if not observations:
        raise ValueError("Reference rollout contains no observations")
    obs = torch.tensor(observations, dtype=torch.float32, device="cpu")
    if not bool(torch.isfinite(obs).all()):
        raise ValueError("Reference rollout contains non-finite observations")
    baseline = FrozenActor(args.baseline_checkpoint.expanduser().resolve(strict=True), "cpu")
    candidate = FrozenActor(args.candidate_checkpoint.expanduser().resolve(strict=True), "cpu")
    with torch.inference_mode():
        expected = torch.cat([baseline(batch) for batch in obs.split(256)])
        actual = torch.cat([candidate(batch) for batch in obs.split(256)])
    metric = _metric(expected.tolist(), actual.tolist())
    metric["mean_abs"] = float((expected - actual).abs().mean())
    report = {"pass": metric["rmse"] <= args.max_action_rmse,
              "reference": str(args.reference), "observations": len(observations),
              "observation_dim": obs.shape[-1], "action_dim": expected.shape[-1],
              "baseline_checkpoint_sha256": baseline.checkpoint_sha256,
              "candidate_checkpoint_sha256": candidate.checkpoint_sha256,
              "max_action_rmse": args.max_action_rmse, "action": metric,
              "inference": "FrozenActor, CPU, identical reference policy observations"}
    if args.out:
        _write_json(args.out, report)
    print(json.dumps(report, indent=2, allow_nan=False))
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    operations = {"capture": capture, "compare": compare, "policy-compare": policy_compare}
    if len(sys.argv) < 2 or sys.argv[1] not in operations:
        raise SystemExit("Usage: manager_regression.py capture|compare|policy-compare [options]")
    operation = operations[sys.argv[1]]
    raise SystemExit(operation(sys.argv[2:]))
