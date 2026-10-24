"""Record each motor-driven motion and uncut combined trials from boot crouch."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.reference_record_plan import recording_plan
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--models', type=Path, required=True)
parser.add_argument('--out', type=Path, required=True)
parser.add_argument('--allow-preview', action='store_true')
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
args.enable_cameras = True
app = AppLauncher(args).app

import gymnasium as gym
import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw
import torch
from isaaclab_tasks.utils import parse_env_cfg
import wheel_humanoid_lab.tasks
from skills.policy import FrozenActor
from skills.reference_web_runtime import CONTRACT, read_json


def slow_kneel_metrics(command):
    """Read the same persistent anti-hop monitor used by training/evaluation."""
    tracker = getattr(command, 'slow_kneel', None)
    if tracker is None:
        return {'monitor_available': False, 'failed': None}
    result = {'monitor_available': True}
    for name in ('failed', 'contact_fraction', 'max_bilateral_loss_s', 'max_air_s',
                 'max_descent_mps', 'max_fast_descent_s', 'max_hop_s'):
        value = getattr(tracker, name, None)
        result[name] = None if value is None else bool(value[0]) if name == 'failed' else float(value[0])
    return result


env = None
results = {}
args.out.mkdir(parents=True, exist_ok=True)
try:
    cfg = parse_env_cfg('Isaac-Q1-Reference-Web-v0', device=args.device or 'cuda:0', num_envs=1)
    cfg.seed = 9127
    cfg.viewer.resolution = (960, 640)
    cfg.sim.render_interval = 8
    env = gym.make('Isaac-Q1-Reference-Web-v0', cfg=cfg, render_mode='rgb_array').unwrapped
    command = env.command_manager.get_term('skill')
    actors, metadata = {}, {}
    for skill in ('standup', 'kneel', 'walk', 'backflip', 'skate'):
        folder = args.models / skill
        path = folder / 'model.pt' if (folder / 'model.pt').exists() else folder / 'latest.pt'
        meta = read_json(folder / ('validated.json' if path.name == 'model.pt' else 'latest.json'))
        validated = path.name == 'model.pt' and meta.get('validated') is True
        if not path.exists() or (not validated and not args.allow_preview):
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        expected = dict(contract=CONTRACT, skill=skill, sha256=digest,
                        reference_sha256=command.trajectory.sha256, urdf_sha256=command.trajectory.urdf_sha256)
        if any(meta.get(key) != value for key, value in expected.items()):
            continue
        actor = FrozenActor(path, env.device)
        if actor.input_dim != 90 or actor.mlp[-1].out_features != 23:
            raise ValueError('Recording policy shape mismatch')
        actors[skill] = actor
        metadata[skill] = dict(meta, validated=validated)

    dt = float(env.step_dt)
    frame_stride = max(1, round(1. / (25. * dt)))
    trace_stride = max(1, round(.5 / dt))
    kneel_duration = float(getattr(command, 'kneel_duration', 24.))
    for clip in recording_plan(kneel_duration):
        # This is the only physical reset in a clip. Failed preparation stays
        # visible; no standing-state reset or second attempt can replace it.
        env.reset(seed=9127)
        command.level = 1.
        command.hold_pose('kneel')
        current, first_failure, failure_reasons = 'kneel', None, []
        trace, selected, requests = [], {'kneel'}, []
        slow_kneel_active = False
        event_steps = clip.event_steps(dt)
        total = math.ceil(clip.duration_s / dt)
        screenshot_steps = {0, (total // 2 // frame_stride) * frame_stride,
                            ((total - 1) // frame_stride) * frame_stride}
        writer = imageio.get_writer(str(args.out / (clip.name + '.mp4')),
                                    fps=1. / (frame_stride * dt), codec='libx264', quality=7, macro_block_size=16)
        try:
            with torch.inference_mode():
                for step in range(total):
                    mode = event_steps.get(step)
                    vx = clip.vx(step * dt)
                    if mode:
                        robot = env.scene['robot']
                        before_root, before_q = robot.data.root_state_w.clone(), robot.data.joint_pos.clone()
                        if mode == 'stand_hold':
                            command.hold_pose('stand')
                            current = 'standup'
                        else:
                            command.set_mode(mode)
                            current = 'skate' if mode == 'slide' else mode
                        slow_kneel_active = mode == 'kneel'
                        if not torch.equal(before_root, robot.data.root_state_w) or not torch.equal(before_q, robot.data.joint_pos):
                            raise RuntimeError('Recording handover wrote the physical robot pose')
                        requests.append(dict(step=step, time_s=step * dt, mode=mode, has_policy=current in actors))
                        selected.add(current)
                    velocity = env.command_manager.get_term('base_velocity')
                    velocity.command[:] = torch.tensor([vx, 0., 0.], device=env.device)
                    velocity.time_left.fill_(1.e9)
                    velocity.is_standing_env.fill_(vx == 0.)
                    command._update_command()
                    observation = env.observation_manager.compute()['policy']
                    action = actors[current](observation) if current in actors else torch.zeros(1, 23, device=env.device)
                    env.step(action)
                    command.update_state()
                    robot = env.scene['robot'].data
                    z = float(robot.root_pos_w[0, 2] - env.scene.env_origins[0, 2])
                    up = float(-robot.projected_gravity_b[0, 2])
                    finite = bool(torch.isfinite(robot.root_state_w).all() and torch.isfinite(robot.joint_pos).all())
                    normal = command.sensor.data.net_forces_w[0, command.support_sensor_ids, 2].clamp_min(0.)
                    wheel_normal = command.sensor.data.net_forces_w[0, command.wheel_sensor_ids, 2].clamp_min(0.)
                    side_normal = [float(normal[0] + normal[2]), float(normal[1] + normal[3])]
                    slow = slow_kneel_metrics(command)
                    reasons = []
                    if not finite: reasons.append('nonfinite_state')
                    if bool(command.illegal[0]): reasons.append('illegal_contact')
                    if z < .14: reasons.append('root_below_floor_limit')
                    if current != 'backflip' and up < .45: reasons.append('not_upright')
                    if slow_kneel_active and slow.get('failed') is True: reasons.append('slow_kneel_contact_or_speed_violation')
                    if slow_kneel_active and not slow.get('monitor_available'): reasons.append('slow_kneel_monitor_missing')
                    if reasons and first_failure is None:
                        first_failure, failure_reasons = step, reasons
                    preview = metadata.get(current, {}).get('validated') is not True
                    if step % trace_stride == 0 or mode or (reasons and first_failure == step):
                        trace.append(dict(step=step, time_s=step * dt, skill=current, preview=preview, height=z, up=up,
                                          vx=float(robot.root_lin_vel_b[0, 0]), root_vz_w_m_s=float(robot.root_lin_vel_w[0, 2]),
                                          support_normal_n=normal.tolist(), support_side_normal_n=side_normal,
                                          bilateral_support=all(value > 12. for value in side_normal), wheel_normal_n=wheel_normal.tolist(),
                                          bilateral_wheel_contact=bool((wheel_normal > 8.).all()),
                                          slow_kneel_active=slow_kneel_active, slow_kneel=slow, first_failure_step=first_failure,
                                          success=bool(command.success[0]), cycles=command.cycles.cycles[0].tolist()))
                    if step % frame_stride == 0:
                        if finite:
                            position = robot.root_pos_w[0].cpu().numpy()
                            target = np.array([position[0], position[1], max(.45, position[2])])
                            env.sim.set_camera_view(target + np.array([2.7, -3., 1.4]), target)
                        frame = Image.fromarray(env.render())
                        draw = ImageDraw.Draw(frame)
                        draw.rectangle((0, 0, 960, 104), fill='black')
                        draw.text((10, 6), 'NEW Q1 | MediaPipe 15-23s | ' + clip.name, fill='white')
                        draw.text((10, 25), f'{step * dt:.2f}s | {current} | ' + ('UNVALIDATED PREVIEW / NOT SUCCESS MODEL' if preview else 'VALIDATED POLICY / continuous trial'), fill='yellow')
                        draw.text((10, 45), f'FAILED at {first_failure * dt:.2f}s / no resets' if first_failure is not None else 'FIRST TRIAL / no hidden resets', fill='red' if first_failure is not None else 'white')
                        draw.text((10, 64), f'z={z:.3f}m up={up:.3f} vx={float(robot.root_lin_vel_b[0, 0]):+.2f} vz={float(robot.root_lin_vel_w[0, 2]):+.2f} m/s', fill='white')
                        draw.text((10, 83), f'wheel L/R={float(wheel_normal[0]):.0f}/{float(wheel_normal[1]):.0f} N | slow-kneel violation={slow.get("failed")}', fill='white')
                        writer.append_data(np.asarray(frame))
                        if step in screenshot_steps:
                            frame.save(args.out / (clip.name + f'_{step:04d}.png'))
        finally:
            writer.close()
        results[clip.name] = dict(focus_skill=clip.focus_skill, duration_s=clip.duration_s, simulation_step_s=dt,
                                  kneel_duration_s=kneel_duration, boot_pose='video_kneel', physical_resets=1,
                                  no_hidden_resets=True, first_failure_step=first_failure, first_failure_reasons=failure_reasons,
                                  requests=requests, trace=trace, models={key: metadata.get(key) for key in selected},
                                  missing_policies=sorted(selected - set(actors)),
                                  all_selected_models_validated=all(metadata.get(key, {}).get('validated') is True for key in selected),
                                  success_claimed=False)
        (args.out / 'review.json').write_text(json.dumps(results, indent=2) + '\n')
finally:
    if env is not None:
        env.close()
    app.close()
