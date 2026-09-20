"""Play a trained skateboard PPO policy in Isaac Sim with WASD keyboard control."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

import cli_args  # isort: skip

parser = argparse.ArgumentParser(description="Play a skateboard PPO policy in Isaac Sim.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during play.")
parser.add_argument("--video_length", type=int, default=500, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default="Isaac-WheelHumanoid-Skateboard-Play-v0", help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
parser.add_argument("--max_steps", type=int, default=None, help="Stop after this many policy steps (headless verify).")
parser.add_argument("--cmd_vx", type=float, default=None, help="Fixed body vx command for headless verify.")
parser.add_argument("--cmd_vy", type=float, default=None, help="Fixed body vy command for headless verify.")
parser.add_argument("--cmd_yaw", type=float, default=None, help="Fixed yaw-rate command for headless verify.")
parser.add_argument(
    "--no-keyboard",
    action="store_true",
    default=False,
    help="Disable WASD teleop and use the env velocity command sampler.",
)
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
if args_cli.video:
    args_cli.enable_cameras = True

sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import math
import os
import time
import weakref

import gymnasium as gym
import torch
from rsl_rl.runners import DistillationRunner, OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper, export_policy_as_jit, export_policy_as_onnx
from isaaclab_rl.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import wheel_humanoid_lab.tasks  # noqa: F401


def _wrap_to_pi(angle: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(angle), torch.cos(angle))


class WasdSkateTeleop:
    """WASD world-frame skate teleop: the robot turns to face the key direction and rolls forward.

    Keys:
        W/S : world +X / -X
        A/D : world +Y / -Y (left / right)
        Q/E : extra yaw in place
        SPACE : stop
        R : reset episode
        L : clear held keys
    """

    def __init__(self, device: str, vx: float = 1.5, vy: float = 1.5, yaw: float = 1.8):
        import carb
        import omni

        self.device = device
        self.vx = vx
        self.vy = vy
        self.yaw = yaw
        self._pressed: set[str] = set()
        self.reset_requested = False
        self._carb = carb
        self._appwindow = omni.appwindow.get_default_app_window()
        self._input = carb.input.acquire_input_interface()
        self._keyboard = self._appwindow.get_keyboard()
        self._keyboard_sub = self._input.subscribe_to_keyboard_events(
            self._keyboard,
            lambda event, *args, obj=weakref.proxy(self): obj._on_event(event, *args),
        )

    def __del__(self):
        try:
            self._input.unsubscribe_to_keyboard_events(self._keyboard, self._keyboard_sub)
        except Exception:
            pass

    def _on_event(self, event, *args, **kwargs):
        raw = event.input
        name = raw.name if hasattr(raw, "name") else str(raw)
        if event.type == self._carb.input.KeyboardEventType.KEY_PRESS:
            if name == "R":
                self.reset_requested = True
            elif name in {"L", "SPACE"}:
                self._pressed.clear()
            else:
                self._pressed.add(name)
        elif event.type == self._carb.input.KeyboardEventType.KEY_RELEASE:
            self._pressed.discard(name)
        return True

    def command(self, heading_w: torch.Tensor) -> torch.Tensor:
        """Return (N, 3) body-frame velocity command (vx, vy, yaw_rate)."""
        wx = (float("W" in self._pressed) - float("S" in self._pressed)) * self.vx
        wy = (float("A" in self._pressed) - float("D" in self._pressed)) * self.vy
        wz = (float("Q" in self._pressed) - float("E" in self._pressed)) * self.yaw

        c = torch.cos(heading_w)
        s = torch.sin(heading_w)
        bx = c * wx + s * wy
        by = -s * wx + c * wy
        if abs(wz) < 1e-6 and (wx * wx + wy * wy) > 0.04:
            target = math.atan2(wy, wx)
            err = _wrap_to_pi(torch.full_like(heading_w, target) - heading_w)
            wz_t = torch.clamp(2.0 * err, min=-self.yaw, max=self.yaw)
        else:
            wz_t = torch.full_like(heading_w, wz)
        return torch.stack((bx.expand_as(heading_w), by.expand_as(heading_w), wz_t), dim=-1)


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Play with RSL-RL agent."""
    task_name = args_cli.task.split(":")[-1]
    train_task_name = task_name.replace("-Play", "")

    agent_cfg: RslRlBaseRunnerCfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("rsl_rl", train_task_name)
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
    elif args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
        ckpt_name = "q1_skate_ppo.pt" if "Q1-Skate" in args_cli.task else "skateboard_ppo.pt"
        default_ckpt = os.path.join(project_root, "checkpoints", ckpt_name)
        if os.path.isfile(default_ckpt):
            resume_path = default_ckpt
        else:
            resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    log_dir = os.path.dirname(resume_path)
    env_cfg.log_dir = log_dir
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during play.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    runner.load(resume_path)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    try:
        policy_nn = runner.alg.policy
    except AttributeError:
        policy_nn = runner.alg.actor_critic

    if hasattr(policy_nn, "actor_obs_normalizer"):
        normalizer = policy_nn.actor_obs_normalizer
    elif hasattr(policy_nn, "student_obs_normalizer"):
        normalizer = policy_nn.student_obs_normalizer
    else:
        normalizer = None

    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    export_policy_as_jit(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.pt")
    export_policy_as_onnx(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.onnx")
    print(f"[INFO] Exported JIT/ONNX policies to: {export_model_dir}")

    dt = env.unwrapped.step_dt
    teleop = None
    if not args_cli.no_keyboard and not args_cli.headless:
        teleop = WasdSkateTeleop(device=str(env.unwrapped.device))
        print(
            "\nWASD skateboard teleop\n"
            "  W/S : slide world +X / -X\n"
            "  A/D : slide world +Y / -Y (left / right)\n"
            "  Q/E : yaw in place\n"
            "  SPACE/L : stop\n"
            "  R : reset episode\n"
        )

    obs = env.get_observations()
    timestep = 0
    start_xy = env.unwrapped.scene["robot"].data.root_pos_w[:, :2].clone()
    while simulation_app.is_running():
        start_time = time.time()
        with torch.inference_mode():
            if teleop is not None:
                robot = env.unwrapped.scene["robot"]
                cmd = teleop.command(robot.data.heading_w)
                vel_term = env.unwrapped.command_manager.get_term("base_velocity")
                vel_term.vel_command_b[:] = cmd
                vel_term.is_standing_env[:] = False
                if teleop.reset_requested:
                    teleop.reset_requested = False
                    env.unwrapped.episode_length_buf[:] = env.unwrapped.max_episode_length
            elif args_cli.cmd_vx is not None or args_cli.cmd_vy is not None or args_cli.cmd_yaw is not None:
                vel_term = env.unwrapped.command_manager.get_term("base_velocity")
                if args_cli.cmd_vx is not None:
                    vel_term.vel_command_b[:, 0] = args_cli.cmd_vx
                if args_cli.cmd_vy is not None:
                    vel_term.vel_command_b[:, 1] = args_cli.cmd_vy
                if args_cli.cmd_yaw is not None:
                    vel_term.vel_command_b[:, 2] = args_cli.cmd_yaw
                vel_term.is_standing_env[:] = False
            actions = policy(obs)
            obs, _, dones, _ = env.step(actions)
            if hasattr(policy_nn, "reset"):
                policy_nn.reset(dones)
        timestep += 1
        if args_cli.video and timestep == args_cli.video_length:
            break
        if args_cli.max_steps is not None and timestep >= args_cli.max_steps:
            break
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    robot = env.unwrapped.scene["robot"]
    end_xy = robot.data.root_pos_w[:, :2]
    delta = (end_xy - start_xy)[0].detach().cpu().tolist()
    height = robot.data.root_pos_w[0, 2].item()
    body_v = robot.data.root_lin_vel_b[0, :2].detach().cpu().tolist()
    heading = robot.data.heading_w[0].item()
    print(
        f"[INFO] Played {timestep} steps. env0 displacement xy=({delta[0]:.2f}, {delta[1]:.2f}) m, "
        f"height={height:.2f} m, body_v=({body_v[0]:.2f}, {body_v[1]:.2f}) m/s, heading={heading:.2f} rad"
    )
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
