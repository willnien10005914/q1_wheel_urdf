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
parser.add_argument("--task", type=str, default="Isaac-Q1-Skate-Play-v0", help="Name of the task.")
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
parser.add_argument(
    "--hold_heading",
    type=float,
    default=None,
    help="World heading (rad) to hold: yaw_cmd = clip(k * wrap(target - heading), +/-0.5). Overrides --cmd_yaw.",
)
parser.add_argument(
    "--cmd_profile",
    type=str,
    default=None,
    help="Piecewise-constant vx schedule 'step:vx,step:vx,...' (e.g. '0:0.0,100:0.6,300:1.2'); overrides --cmd_vx.",
)
parser.add_argument("--cmd_vy", type=float, default=None, help="Fixed body vy command for headless verify.")
parser.add_argument("--cmd_yaw", type=float, default=None, help="Fixed yaw-rate command for headless verify.")
parser.add_argument(
    "--no-keyboard",
    action="store_true",
    default=False,
    help="Disable WASD teleop and use the env velocity command sampler.",
)
parser.add_argument("--web-port", type=int, default=8766, help="Port for the live motor web UI.")
parser.add_argument(
    "--posture-checkpoint",
    type=str,
    default=None,
    help="Kneel/stand posture PPO (default checkpoints/q1_posture_ppo.pt; scripted fallback if missing).",
)
parser.add_argument(
    "--slide-checkpoint",
    type=str,
    default=None,
    help="Push-skate slide PPO (default checkpoints/q1_slide_ppo.pt; skate PPO wanders if missing).",
)
parser.add_argument(
    "--no-web",
    action="store_true",
    default=False,
    help="Do not start the motor web UI while playing.",
)
parser.add_argument(
    "--no-browser",
    action="store_true",
    default=False,
    help="Start the web UI but do not open a browser tab.",
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
import threading
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
from play_modes import LoadedPolicy, ModeController
from play_web import start_play_web


def _keep_timeline_playing() -> None:
    """Space in Isaac Sim pauses the Kit timeline; play.py then blocks inside sim.step().

    Keep the timeline running so WASD/R and physics stay in lockstep. Pause still zeros the
    robot command; it must not freeze the sim.
    """
    try:
        import omni.timeline

        timeline = omni.timeline.get_timeline_interface()
        if timeline.is_stopped():
            return
        if not timeline.is_playing():
            timeline.play()
    except Exception:
        pass


def _start_timeline_watchdog() -> None:
    """Unpause even when the main loop is blocked inside sim.step()."""

    def _loop() -> None:
        while True:
            time.sleep(0.05)
            _keep_timeline_playing()

    threading.Thread(target=_loop, name="q1-timeline-watchdog", daemon=True).start()


def _hard_reset(env, policy_nn) -> object:
    """Respawn at the skate keyframe instead of the timeout-hack that made the robot vanish."""
    unwrapped = env.unwrapped
    unwrapped.reset()
    robot = unwrapped.scene["robot"]
    root = robot.data.default_root_state.clone()
    root[:, :3] += unwrapped.scene.env_origins
    robot.write_root_pose_to_sim(root[:, :7])
    robot.write_root_velocity_to_sim(torch.zeros(unwrapped.num_envs, 6, device=unwrapped.device))
    robot.write_joint_state_to_sim(robot.data.default_joint_pos, robot.data.default_joint_vel)
    unwrapped.scene.write_data_to_sim()
    unwrapped.sim.forward()
    vel_term = unwrapped.command_manager.get_term("base_velocity")
    vel_term.vel_command_b[:] = 0.0
    vel_term.is_standing_env[:] = True
    if hasattr(policy_nn, "reset"):
        policy_nn.reset(torch.ones(unwrapped.num_envs, dtype=torch.bool, device=unwrapped.device))
    _keep_timeline_playing()
    return env.get_observations()


def _named_joints(robot) -> dict[str, float]:
    q = robot.data.joint_pos[0].detach().cpu()
    return {name: float(q[i]) for i, name in enumerate(robot.joint_names)}


def _named_defaults(robot) -> dict[str, float]:
    q = robot.data.default_joint_pos[0].detach().cpu()
    return {name: float(q[i]) for i, name in enumerate(robot.joint_names)}


# Scripted FALLBACK only (used when checkpoints/q1_posture_ppo.pt does not exist yet). PD targets
# measured in sim to rest on four wheels: the knee sags ~0.1 rad under load, so the target is below
# the ~2.35 rad the joint actually settles at. Open-loop, so it is not robust - the posture PPO is.
_KNEEL_PATCH = {
    "l_hip_pitch_joint": -1.00,
    "r_hip_pitch_joint": -1.00,
    "l_hip_roll_joint": 0.0,
    "r_hip_roll_joint": 0.0,
    "l_knee_joint": 2.25,
    "r_knee_joint": 2.25,
    "waist_pitch_joint": 0.35,
    "waist_yaw_joint": 0.0,
    "waist_roll_joint": 0.0,
}


# Fallback squat must stay slow enough that the foot wheels never leave the ground.
_KNEEL_DURATION_S = 3.0
_STAND_DURATION_S = 3.2


class ScriptedKneelStand:
    """Fallback: interpolate hip/knee between skate and four-wheel kneel when no posture PPO exists."""

    def __init__(self) -> None:
        self.mode = "ppo"  # ppo | kneel_motion | kneel_hold | stand_motion
        self.latest: dict[str, float] | None = None
        self._t = 0.0
        self._duration = 1.0
        self._start: dict[str, float] = {}
        self._end: dict[str, float] = {}

    def reset(self) -> None:
        self.mode = "ppo"
        self.latest = None
        self._t = 0.0
        self._start = {}
        self._end = {}

    def request_kneel(self, current: dict[str, float], defaults: dict[str, float]) -> bool:
        if self.mode in {"kneel_motion", "kneel_hold"}:
            return False
        self._start = dict(current)
        self._end = {**defaults, **_KNEEL_PATCH}
        self._t = 0.0
        self._duration = _KNEEL_DURATION_S
        self.mode = "kneel_motion"
        return True

    def request_stand(self, current: dict[str, float], defaults: dict[str, float]) -> bool:
        if self.mode == "stand_motion":
            return False
        self._start = dict(current)
        self._end = dict(defaults)
        self._t = 0.0
        self._duration = _STAND_DURATION_S
        self.mode = "stand_motion"
        return True

    def step(self, dt: float) -> tuple[dict[str, float] | None, str | None]:
        """Return (joint overrides or None, event). Wheel keys are velocity targets (0)."""
        if self.mode == "ppo":
            self.latest = None
            return None, None
        event = None
        if self.mode in {"kneel_motion", "stand_motion"}:
            self._t += dt
            u = min(1.0, self._t / max(self._duration, 1e-3))
            s = u * u * (3.0 - 2.0 * u)
            pose = {}
            keys = set(self._start) | set(self._end)
            for name in keys:
                if name.endswith("_wheel_joint") or name.endswith("_roller_joint"):
                    continue
                a = self._start.get(name, self._end.get(name, 0.0))
                b = self._end.get(name, a)
                pose[name] = a + (b - a) * s
            pose["l_wheel_joint"] = 0.0
            pose["r_wheel_joint"] = 0.0
            self.latest = pose
            if u >= 1.0:
                if self.mode == "kneel_motion":
                    self.mode = "kneel_hold"
                    event = "knelt"
                else:
                    # Keep this frame's skate targets so the last apply isn't PPO.
                    self.mode = "ppo"
                    event = "stood"
            return pose, event
        pose = {k: v for k, v in self._end.items() if not k.endswith("_roller_joint")}
        pose["l_wheel_joint"] = 0.0
        pose["r_wheel_joint"] = 0.0
        self.latest = pose
        return pose, None

    @property
    def blocking(self) -> bool:
        # Stay blocking on the last stand frame while skate targets are still applied.
        return self.mode != "ppo" or self.latest is not None

    @property
    def label(self) -> str:
        return {
            "ppo": "ppo",
            "kneel_motion": "kneeling",
            "kneel_hold": "kneel",
            "stand_motion": "standing",
        }[self.mode]


class WasdSkateTeleop:
    """Body-frame teleop that matches how the skate PPO was trained.

    Training never commanded lateral velocity (vy = 0). Strafe (world-Y / body-Y) is out of
    distribution and tips the robot. A/D therefore yaw in place; W/S roll along the heading
    with a slow reverse limit.
    """

    def __init__(
        self,
        device: str,
        vx_fwd: float = 1.2,
        vx_rev: float = 0.35,
        yaw: float = 0.40,
    ):
        import carb
        import omni

        self.device = device
        self.vx_fwd = vx_fwd
        self.vx_rev = vx_rev
        self.yaw = yaw
        self._pressed: set[str] = set()
        self.reset_requested = False
        self.kneel_requested = False
        self.stand_requested = False
        self.slide_toggle_requested = False
        self._cmd = torch.zeros(3, device=device)
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
            if name in {"R", "HOME"}:
                self.reset_requested = True
            elif name in {"K"}:
                self.kneel_requested = True
            elif name in {"U"}:
                self.stand_requested = True
            elif name in {"P"}:
                self.slide_toggle_requested = True
            elif name in {"L", "SPACE", "X"}:
                self._pressed.clear()
                self._cmd.zero_()
                _keep_timeline_playing()
            else:
                self._pressed.add(name)
        elif event.type == self._carb.input.KeyboardEventType.KEY_RELEASE:
            self._pressed.discard(name)
        return True

    def target(self) -> tuple[float, float, float]:
        vx = 0.0
        yaw = 0.0
        if "W" in self._pressed:
            vx += self.vx_fwd
        if "S" in self._pressed:
            vx -= self.vx_rev
        if "A" in self._pressed:
            yaw += self.yaw
        if "D" in self._pressed:
            yaw -= self.yaw
        if "Q" in self._pressed:
            yaw += self.yaw * 1.25
        if "E" in self._pressed:
            yaw -= self.yaw * 1.25
        return vx, 0.0, yaw

    def command(self, n_envs: int, dt: float) -> torch.Tensor:
        tx, ty, tz = self.target()
        reversing = tx < 0.0 or float(self._cmd[0]) < 0.0
        tau_x = 0.70 if reversing else 0.35
        alpha_x = 1.0 - math.exp(-dt / tau_x)
        alpha_z = 1.0 - math.exp(-dt / 0.45)
        self._cmd[0] += (tx - self._cmd[0]) * alpha_x
        self._cmd[1] = ty
        self._cmd[2] += (tz - self._cmd[2]) * alpha_z
        return self._cmd.unsqueeze(0).expand(n_envs, -1).clone()

    def zero(self) -> None:
        self._pressed.clear()
        self._cmd.zero_()

    @property
    def any_pressed(self) -> bool:
        return bool(self._pressed & {"W", "A", "S", "D", "Q", "E"})


def _write_processed_overrides(mgr, overrides: dict[str, float]) -> None:
    if not overrides:
        return
    for name in mgr.active_terms:
        term = mgr.get_term(name)
        joint_names = getattr(term, "_joint_names", None)
        processed = getattr(term, "_processed_actions", None)
        if joint_names is None or processed is None:
            continue
        for i, joint in enumerate(joint_names):
            if joint in overrides:
                processed[:, i] = float(overrides[joint])


def _install_override_hook(
    env, web_state, pose_ctrl: ScriptedKneelStand | None = None, mode_ctrl: ModeController | None = None
) -> None:
    """Rewrite the joint targets after PPO processing and again every physics apply.

    Priority: posture-PPO remap (wider leg clip) < web slider overrides < scripted kneel fallback.
    Fall/contact episode resets are suppressed while a posture transition runs: the kneel drops the
    pelvis to ~0.45 m (the skate PLAY env terminates below 0.60 m), which otherwise snapped the robot
    back to the skate pose mid-kneel - the kneel/stand loop seen before.
    """
    unwrapped = env.unwrapped
    mgr = unwrapped.action_manager
    orig_process = mgr.process_action
    orig_apply = mgr.apply_action
    pos_term = mgr.get_term("joint_pos")

    def _overrides() -> dict[str, float]:
        overrides: dict[str, float] = {}
        if web_state is not None:
            overrides.update(web_state.pop_overrides())
        if pose_ctrl is not None and pose_ctrl.latest:
            overrides.update(pose_ctrl.latest)
        return overrides

    def _apply_all(include_web: bool) -> None:
        if mode_ctrl is not None and mode_ctrl.processed_override is not None:
            pos_term._processed_actions[:] = mode_ctrl.processed_override
        if include_web:
            _write_processed_overrides(mgr, _overrides())
        elif pose_ctrl is not None and pose_ctrl.latest:
            _write_processed_overrides(mgr, pose_ctrl.latest)

    def hooked_process(action: torch.Tensor) -> None:
        orig_process(action)
        _apply_all(include_web=True)

    def hooked_apply() -> None:
        # Re-apply every physics substep so the skate clip cannot sneak back in.
        _apply_all(include_web=False)
        orig_apply()

    mgr.process_action = hooked_process
    mgr.apply_action = hooked_apply

    tm = unwrapped.termination_manager
    orig_compute = tm.compute

    def hooked_compute(*args, **kwargs):
        reset_buf = orig_compute(*args, **kwargs)
        suppress = (pose_ctrl is not None and pose_ctrl.blocking) or (
            mode_ctrl is not None and mode_ctrl.suppress_terminations
        )
        if not suppress:
            return reset_buf
        tm._truncated_buf[:] = False
        tm._terminated_buf[:] = False
        return tm._truncated_buf | tm._terminated_buf

    tm.compute = hooked_compute


def _publish_web_state(
    env, web_state, cmd: torch.Tensor, pose_label: str = "ppo", extra: dict | None = None
) -> None:
    extra = extra or {}
    unwrapped = env.unwrapped
    robot = unwrapped.scene["robot"]
    names = list(robot.joint_names)
    pos = robot.data.joint_pos[0].detach().cpu()
    vel = robot.data.joint_vel[0].detach().cpu()
    tau = robot.data.applied_torque[0].detach().cpu() if robot.data.applied_torque is not None else torch.zeros_like(pos)
    root = robot.data.root_pos_w[0].detach().cpu()
    body_v = robot.data.root_lin_vel_b[0].detach().cpu()
    heading = float(robot.data.heading_w[0].item())
    cmd0 = cmd[0].detach().cpu()

    targets: dict[str, float] = {}
    actions: dict[str, float] = {}
    mgr = unwrapped.action_manager
    for term_name in mgr.active_terms:
        term = mgr.get_term(term_name)
        joint_names = getattr(term, "_joint_names", None)
        processed = getattr(term, "_processed_actions", None)
        raw = getattr(term, "_raw_actions", None)
        if joint_names is None:
            continue
        for i, joint in enumerate(joint_names):
            if processed is not None:
                targets[joint] = float(processed[0, i].item())
            if raw is not None:
                actions[joint] = float(raw[0, i].item())

    joints = {}
    for i, name in enumerate(names):
        joints[name] = {
            "pos": float(pos[i]),
            "vel": float(vel[i]),
            "tau": float(tau[i]),
            "target": targets.get(name, float(pos[i])),
            "action": actions.get(name, 0.0),
            "wheel": name.endswith("_wheel_joint"),
        }
    web_state.publish(
        {
            "ok": True,
            "t": float(unwrapped.episode_length_buf[0].item()) * float(unwrapped.step_dt),
            "cmd": {"vx": float(cmd0[0]), "vy": float(cmd0[1]), "yaw": float(cmd0[2])},
            "pose": pose_label,
            "policies": extra.get("policies") or {},
            "train": extra.get("train") or {},
            "base": {
                "x": float(root[0]),
                "y": float(root[1]),
                "z": float(root[2]),
                "heading": heading,
                "vx": float(body_v[0]),
                "vy": float(body_v[1]),
            },
            "joints": joints,
        }
    )


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
    _start_timeline_watchdog()
    teleop = None
    if not args_cli.no_keyboard and not args_cli.headless:
        teleop = WasdSkateTeleop(device=str(env.unwrapped.device))
        print(
            "\nWASD skate teleop (body frame, matches PPO training)\n"
            "  W     : roll forward (vx ≈ 1.2 m/s, ramped)\n"
            "  S     : slow reverse (vx ≈ -0.35 m/s; policy barely trained reverse)\n"
            "  A/D   : yaw left / right  (NOT strafe — vy is always 0)\n"
            "  Q/E   : extra yaw\n"
            "  SPACE/X/L : stop  (Space no longer pauses Isaac Sim)\n"
            "  K         : kneel (posture PPO if trained, else scripted)\n"
            "  U         : stand up, then skate PPO balances\n"
            "  P         : toggle slide (X2 push-skate, random wander; WASD overrides)\n"
            "  R/Home    : respawn at origin\n"
        )

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    posture_ckpt = args_cli.posture_checkpoint or os.path.join(project_root, "checkpoints", "q1_posture_ppo.pt")
    slide_ckpt = args_cli.slide_checkpoint or os.path.join(project_root, "checkpoints", "q1_slide_ppo.pt")
    device = env.unwrapped.device
    mode_ctrl = ModeController(
        env,
        {
            "skate": LoadedPolicy.wrap(env, agent_cfg, resume_path, str(device), runner, policy, policy_nn),
            "posture": LoadedPolicy(env, agent_cfg, posture_ckpt, str(device)),
            "slide": LoadedPolicy(env, agent_cfg, slide_ckpt, str(device)),
        },
        str(device),
    )
    if mode_ctrl.has_posture_policy:
        print(f"[INFO] Kneel/stand posture PPO: {posture_ckpt}")
    else:
        print("[INFO] No q1_posture_ppo.pt yet — K/U uses the scripted squat. Train with ./train_posture.sh")
    if mode_ctrl.has_slide_policy:
        print(f"[INFO] Slide / X2-skate PPO: {slide_ckpt}")
    else:
        print("[INFO] No q1_slide_ppo.pt yet — P wanders with the skate PPO. Train with ./train_slide.sh")

    pose_ctrl = ScriptedKneelStand()
    web_state = None
    if not args_cli.no_web and not args_cli.headless:
        web_state = start_play_web(port=args_cli.web_port, open_browser=not args_cli.no_browser)
        _install_override_hook(env, web_state, pose_ctrl, mode_ctrl)
    else:
        _install_override_hook(env, None, pose_ctrl, mode_ctrl)

    cmd_profile: list[tuple[int, float]] = []
    if args_cli.cmd_profile:
        for knot in args_cli.cmd_profile.split(","):
            step_str, vx_str = knot.split(":")
            cmd_profile.append((int(step_str), float(vx_str)))
        cmd_profile.sort()

    obs = env.get_observations()
    timestep = 0
    start_xy = env.unwrapped.scene["robot"].data.root_pos_w[:, :2].clone()
    n_envs = env.unwrapped.num_envs
    device = env.unwrapped.device
    last_cmd = torch.zeros(n_envs, 3, device=device)
    while simulation_app.is_running():
        start_time = time.time()
        _keep_timeline_playing()
        with torch.inference_mode():
            reset_now = bool(teleop is not None and teleop.reset_requested)
            if web_state is not None and web_state.consume_reset():
                reset_now = True
            if reset_now:
                if teleop is not None:
                    teleop.reset_requested = False
                    teleop.zero()
                pose_ctrl.reset()
                mode_ctrl.reset()
                if web_state is not None:
                    web_state.apply_joint_msg({"release_all": True})
                obs = _hard_reset(env, policy_nn)
                last_cmd.zero_()
                print("[INFO] Reset: robot respawned at origin.")

            robot = env.unwrapped.scene["robot"]
            defaults = _named_defaults(robot)
            current = _named_joints(robot)

            def _dispatch_pose(name: str) -> None:
                if name in {"slide", "skate"}:
                    if pose_ctrl.blocking:
                        print("[INFO] Stand up before changing skate/slide mode.")
                        return
                    msg = mode_ctrl.request(name)
                    if msg:
                        print(f"[INFO] {msg}")
                    return
                if mode_ctrl.has_posture_policy:
                    msg = mode_ctrl.request(name)
                    if msg:
                        pose_ctrl.reset()
                        if teleop is not None:
                            teleop.zero()
                        print(f"[INFO] {msg}")
                    return
                if name == "kneel" and pose_ctrl.request_kneel(current, defaults):
                    if teleop is not None:
                        teleop.zero()
                    print("[INFO] Kneel: scripted squat (train posture PPO with ./train_posture.sh).")
                elif name == "stand" and pose_ctrl.request_stand(current, defaults):
                    print("[INFO] Stand: scripted interpolation, then skate PPO.")

            if teleop is not None and teleop.kneel_requested:
                teleop.kneel_requested = False
                _dispatch_pose("kneel")
            if teleop is not None and teleop.stand_requested:
                teleop.stand_requested = False
                _dispatch_pose("stand")
            if teleop is not None and teleop.slide_toggle_requested:
                teleop.slide_toggle_requested = False
                _dispatch_pose("skate" if mode_ctrl.mode == "slide" else "slide")
            if web_state is not None:
                pose_name = web_state.consume_pose()
                if pose_name:
                    _dispatch_pose(pose_name)

            if mode_ctrl.mode == "posture":
                pose_ctrl.reset()
            else:
                _, pose_event = pose_ctrl.step(dt)
                if pose_event == "knelt":
                    print("[INFO] Kneel hold: four wheels on the ground.")
                elif pose_event == "stood":
                    if web_state is not None:
                        web_state.apply_joint_msg({"release_all": True})
                    if hasattr(policy_nn, "reset"):
                        policy_nn.reset(torch.ones(n_envs, dtype=torch.bool, device=device))
                    print("[INFO] Standing skate pose reached — skate PPO balancing.")

            blocking = pose_ctrl.blocking or mode_ctrl.blocking
            cmd = last_cmd
            if blocking:
                vel_term = env.unwrapped.command_manager.get_term("base_velocity")
                vel_term.vel_command_b[:] = 0.0
                vel_term.is_standing_env[:] = True
                cmd = vel_term.vel_command_b
                last_cmd = cmd
            elif mode_ctrl.mode == "slide" and (teleop is None or not teleop.any_pressed):
                cmd = mode_ctrl.wander.step(dt)
                vel_term = env.unwrapped.command_manager.get_term("base_velocity")
                vel_term.vel_command_b[:] = cmd
                vel_term.is_standing_env[:] = False
            elif teleop is not None:
                cmd = teleop.command(n_envs, dt)
                vel_term = env.unwrapped.command_manager.get_term("base_velocity")
                vel_term.vel_command_b[:] = cmd
                vel_term.is_standing_env[:] = cmd.norm(dim=-1) < 0.05
            elif (
                cmd_profile
                or args_cli.cmd_vx is not None
                or args_cli.cmd_vy is not None
                or args_cli.cmd_yaw is not None
                or args_cli.hold_heading is not None
            ):
                vel_term = env.unwrapped.command_manager.get_term("base_velocity")
                cmd = vel_term.vel_command_b.clone()
                if cmd_profile:
                    vx = cmd_profile[0][1]
                    for knot_step, knot_vx in cmd_profile:
                        if timestep >= knot_step:
                            vx = knot_vx
                    cmd[:, 0] = vx
                elif args_cli.cmd_vx is not None:
                    cmd[:, 0] = args_cli.cmd_vx
                if args_cli.cmd_vy is not None:
                    cmd[:, 1] = args_cli.cmd_vy
                else:
                    cmd[:, 1] = 0.0
                if args_cli.hold_heading is not None:
                    heading_w = env.unwrapped.scene["robot"].data.heading_w
                    err = torch.remainder(args_cli.hold_heading - heading_w + math.pi, 2 * math.pi) - math.pi
                    cmd[:, 2] = torch.clamp(1.0 * err, -0.5, 0.5)
                elif args_cli.cmd_yaw is not None:
                    cmd[:, 2] = args_cli.cmd_yaw
                vel_term.vel_command_b[:] = cmd
                vel_term.is_standing_env[:] = False
            last_cmd = cmd

            if (
                web_state is not None
                and web_state.web_cmd
                and teleop is not None
                and not teleop._pressed
                and not blocking
            ):
                with web_state._lock:
                    wcmd = dict(web_state.web_cmd or {})
                vel_term = env.unwrapped.command_manager.get_term("base_velocity")
                if "vx" in wcmd:
                    vel_term.vel_command_b[:, 0] = wcmd["vx"]
                vel_term.vel_command_b[:, 1] = 0.0
                if "yaw" in wcmd:
                    vel_term.vel_command_b[:, 2] = wcmd["yaw"]
                cmd = vel_term.vel_command_b
                last_cmd = cmd

            if timestep % 50 == 0:
                mode_ctrl.poll_reload()
            if pose_ctrl.blocking and not mode_ctrl.has_posture_policy:
                act_dim = env.unwrapped.action_manager.total_action_dim
                actions = torch.zeros(n_envs, act_dim, device=device)
            else:
                actions = mode_ctrl.act(obs, dt)
            obs, _, dones, _ = env.step(actions)
            handover = mode_ctrl.after_step(dones)
            if handover:
                print(f"[INFO] {handover}")
            elif hasattr(policy_nn, "reset") and not blocking:
                policy_nn.reset(dones)
            if web_state is not None:
                pose_label = mode_ctrl.label if not pose_ctrl.blocking else pose_ctrl.label
                extra = {
                    "policies": {
                        "posture": mode_ctrl.has_posture_policy,
                        "slide": mode_ctrl.has_slide_policy,
                    },
                }
                _publish_web_state(env, web_state, cmd, pose_label, extra)
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
