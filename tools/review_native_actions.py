"""Record honest first-trial native-URDF reviews, including incomplete skills.

No automatic reset is allowed inside a clip. Only env.reset() sets a pose;
switching a skill changes motor/command buffers and is checked for pose writes.
"""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from isaaclab.app import AppLauncher

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--stand-checkpoint', type=Path, required=True)
p.add_argument('--smoke-models', type=Path, required=True)
p.add_argument('--out', type=Path, required=True)
p.add_argument('--seed', type=int, default=9127)
p.add_argument('--cases', nargs='*')
AppLauncher.add_app_launcher_args(p)
a = p.parse_args(); a.headless = True; a.enable_cameras = True
app = AppLauncher(a).app

import gymnasium as gym
import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw
import torch
from isaaclab_tasks.utils import parse_env_cfg
import wheel_humanoid_lab.tasks
from wheel_humanoid_lab.tasks.manager_based.inertia import native_skills as mdp
from skills.policy import FrozenActor

cases = [
    ('stand', 'stand', 600, a.stand_checkpoint, 'VALIDATED STAND PPO / iteration 999'),
    ('walk_controller', 'walk', 700, None, 'NO WALK PPO YET / native motor controller'),
    ('walk_smoke', 'walk', 600, a.smoke_models/'walk/latest.pt', 'SMOKE PPO ONLY / 3 updates / NOT validated'),
    ('kneel', 'kneel', 500, None, 'NO KNEEL PPO / native motor controller'),
    ('standup_controller', 'standup', 600, None, 'NO STANDUP PPO YET / native motor controller'),
    ('standup_smoke', 'standup', 600, a.smoke_models/'standup/latest.pt', 'SMOKE PPO ONLY / 3 updates / NOT validated'),
    ('backflip_smoke', 'backflip', 350, a.smoke_models/'backflip/latest.pt', 'SMOKE PPO ONLY / 3 updates / NOT validated'),
    ('kneel_to_standup', 'stand', 850, None, 'END-TO-END CONTROLLER / NO STANDUP PPO'),
]
if a.cases:
    cases = [c for c in cases if c[0] in a.cases]
    if {c[0] for c in cases} != set(a.cases): p.error('Unknown case')
a.out.mkdir(parents=True, exist_ok=True)
env = None
results = json.loads((a.out/'review.json').read_text()) if (a.out/'review.json').exists() else {}
try:
    cfg = parse_env_cfg('Isaac-Q1-Native-Web-v0', device=a.device or 'cuda:0', num_envs=1)
    cfg.seed = a.seed
    cfg.viewer.resolution = (960, 640)
    cfg.sim.render_interval = 20
    env = gym.make('Isaac-Q1-Native-Web-v0', cfg=cfg, render_mode='rgb_array').unwrapped
    s = env.command_manager.get_term('skill')
    with torch.inference_mode():
        for name, mode, steps, checkpoint, label in cases:
            actor = FrozenActor(checkpoint, env.device) if checkpoint else None
            if actor: assert actor.input_dim == 90 and actor.mlp[-1].out_features == 23
            env.cfg.native_skill = 'standup' if mode == 'standup' else 'stand'
            s.cfg.mode = env.cfg.native_skill
            env.reset(seed=a.seed)
            s.level = 1.
            rows = []; first_failure = None; successes = False; lost_feet = 0; task_hold = 0.; best_task_hold = 0.
            writer = imageio.get_writer(str(a.out/f'{name}.mp4'), fps=25, codec='libx264', quality=7, macro_block_size=16)
            try:
                for step in range(steps):
                    target_mode = mode
                    if name == 'kneel': target_mode = 'stand' if step < 100 else 'kneel'
                    if name == 'kneel_to_standup': target_mode = 'stand' if step < 100 else 'kneel' if step < 400 else 'standup'
                    if int(s.mode[0]) != mdp.MODES.index(target_mode):
                        before_root = env.scene['robot'].data.root_state_w.clone()
                        before_q = env.scene['robot'].data.joint_pos.clone()
                        s.set_mode(target_mode)
                        assert torch.equal(before_root, env.scene['robot'].data.root_state_w)
                        assert torch.equal(before_q, env.scene['robot'].data.joint_pos)
                    command = env.command_manager.get_term('base_velocity')
                    vx = yaw = 0.
                    if mode == 'walk':
                        if 100 <= step < 300: vx = .4
                        elif 350 <= step < 450: yaw = .25
                        elif 450 <= step < 550: yaw = -.25
                        elif step >= 550: vx = -.2
                    command.command[:] = torch.tensor([vx, 0., yaw], device=env.device)
                    command.time_left.fill_(1.e9); command.is_standing_env.fill_(not (vx or yaw))
                    s._update_command()
                    obs = env.observation_manager.compute()['policy']
                    action = actor(obs) if actor else torch.zeros(1, 23, device=env.device)
                    assert torch.isfinite(action).all()
                    env.step(action)
                    failure = bool(mdp.physical_failure(env)[0])
                    r = env.scene['robot'].data
                    z = float((r.root_pos_w[:, 2]-env.scene.env_origins[:, 2])[0])
                    upright = float(-r.projected_gravity_b[0, 2])
                    forces = s.sensor.data.net_forces_w.norm(dim=-1)
                    feet_ids, _ = s.sensor.find_bodies(['left_wheel_link', 'right_wheel_link'])
                    feet = (forces[0, feet_ids] > 5.).tolist()
                    if target_mode == 'kneel':
                        # Native physical_failure currently uses standing height
                        # for kneel; apply the actual kneeling height gate here.
                        failure = z < .20 or upright < .5 or bool(s.illegal[0])
                    if failure and first_failure is None: first_failure = step
                    goal_mode = 'standup' if name == 'kneel_to_standup' else mode
                    if goal_mode == 'kneel':
                        achieved = .24 < z < .60 and upright > .85 and any(feet) and not bool(s.illegal[0])
                    else: achieved = bool(s.success[0])
                    achieved = achieved and target_mode == goal_mode and first_failure is None
                    task_hold = task_hold + env.step_dt if achieved else 0.
                    best_task_hold = max(best_task_hold, task_hold)
                    successes |= task_hold >= 1.
                    if step % 100 == 0: print('REVIEW_PROGRESS', name, step, z, flush=True)
                    if target_mode == 'kneel' and not any(feet): lost_feet += 1
                    row = dict(step=step, mode=target_mode, height=z, upright=upright,
                               vx=float(r.root_lin_vel_b[0, 0]), x=float(r.root_pos_w[0, 0]),
                               failure=failure, success=bool(s.success[0]), illegal=bool(s.illegal[0]),
                               winding=float(s.winding[0]), hold_s=float(s.best_hold[0]), feet=feet,
                               command_vx=vx, command_yaw=yaw)
                    if step % 10 == 0: rows.append(row)
                    if step % 2 == 0:
                        pos = r.root_pos_w[0].cpu().numpy()
                        target = np.array([pos[0], pos[1], max(.5, pos[2])])
                        env.sim.set_camera_view(target+np.array([2.6, -3., 1.4]), target)
                        frame = Image.fromarray(env.render())
                        if step == 0:
                            for _ in range(5): frame = Image.fromarray(env.render())
                        draw = ImageDraw.Draw(frame); draw.rectangle((0, 0, 960, 85), fill='black')
                        draw.text((10, 7), f'NEW Q1 URDF | 23 actions / 90 obs | {name}', fill='white')
                        draw.text((10, 26), label, fill='yellow')
                        state = f'FAILED at {first_failure*.02:.2f}s' if first_failure is not None else 'FIRST TRIAL / no resets'
                        draw.text((10, 45), f'{step*.02:.2f}s | {target_mode} | {state} | z={z:.3f}m up={upright:.3f}', fill='white')
                        draw.text((10, 64), f'cmd vx={vx:+.2f} yaw={yaw:+.2f} | actual vx={row["vx"]:+.2f} | x={row["x"]:+.2f}', fill='white')
                        writer.append_data(np.asarray(frame))
                        if step in (0, steps//2, steps-2): frame.save(a.out/f'{name}_{step:04d}.png')
            finally: writer.close()
            results[name] = dict(label=label, mode=mode, seed=a.seed, steps=steps,
                                 checkpoint=str(checkpoint) if checkpoint else None,
                                 checkpoint_sha256=actor.checkpoint_sha256 if actor else None,
                                 checkpoint_iteration=actor.training_iteration if actor else None,
                                 first_failure_step=first_failure, achieved_before_failure=successes,
                                 kneel_no_foot_contact_frames=lost_feet, best_task_hold_s=best_task_hold, final=row, trace=rows,
                                 no_automatic_resets=True, handover_no_pose_writes=True)
            (a.out/'review.json').write_text(json.dumps(results, indent=2)+'\n')
            print('REVIEW_CASE', name, json.dumps({k:v for k,v in results[name].items() if k != 'trace'}), flush=True)
finally:
    if env is not None: env.close()
    app.close()
