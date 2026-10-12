"""Web-controlled Q1 skills on the same motor-limited physical articulation."""
import argparse,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
sys.path[:0]=[str(ROOT),str(ROOT/'q1_backflip_v2')]
from isaaclab.app import AppLauncher
p=argparse.ArgumentParser()
p.add_argument('--web-port',type=int,default=8765);p.add_argument('--no-browser',action='store_true')
p.add_argument('--checkpoint');p.add_argument('--teacher');p.add_argument('--num_envs',type=int,default=1)
p.add_argument('--seed',type=int,default=4107)
p.add_argument('--unbox-checkpoint',help='Explicit landing-policy candidate for validation.')
p.add_argument('--initial-kneel',action='store_true',help='Explicit demo initial state; Unbox itself never resets pose.')
p.add_argument('--steps',type=int,default=0);p.add_argument('--out');p.add_argument('--sequence',default='')
p.add_argument('--real-time',action='store_true')
p.add_argument('--no-keyboard',action='store_true')
p.add_argument('--video-path')
AppLauncher.add_app_launcher_args(p);args=p.parse_args();args.enable_cameras=bool(args.video_path) or args.enable_cameras;app=AppLauncher(args).app
import torch
from skills.interactive_env import Env,Config
from play_web import start_play_web
from skills.teleop import IsaacKeyboard,apply_drive
cfg=Config();cfg.scene.num_envs=args.num_envs;cfg.seed=args.seed
cfg.viewer.origin_type='env';cfg.viewer.env_index=0;cfg.viewer.eye=(2.4,-3.,1.8);cfg.viewer.lookat=(.4,0,.7);cfg.viewer.resolution=(960,720)
cfg.sim.device=args.device or 'cuda:0'
env=Env(cfg,ROOT,args.checkpoint,teacher_path=args.teacher,render_mode='rgb_array' if args.video_path else None)
if args.unbox_checkpoint:
    from skills.policy import FrozenActor
    env.recoveries['landing_rise'].actor=FrozenActor(args.unbox_checkpoint,env.device)
state=start_play_web(port=args.web_port,open_browser=not args.no_browser and not args.headless)
keyboard=IsaacKeyboard() if not args.headless and not args.no_keyboard else None
if keyboard:print('Q1 keyboard: WASD drive, Space stop, R reset, K kneel, U Unbox, B Backflip, P slide. Click the Isaac viewport for focus.',flush=True)
schedule={int(k):v for k,v in (item.split(':',1) for item in args.sequence.split(',') if item)}
records=[];requests=[]
writer=None
if args.video_path:
    import imageio.v2 as imageio
    import numpy as np
    from PIL import Image,ImageDraw
    path=Path(args.video_path);path.parent.mkdir(parents=True,exist_ok=True)
    writer=imageio.get_writer(str(path),fps=50,codec='libx264',quality=8,macro_block_size=16)
env.reset(seed=args.seed)
if args.initial_kneel:
    runtime=env.recoveries['boot_kneel']
    runtime.mdp.reset(runtime.context,env.robot._ALL_INDICES,boot_kneel=True)
    env.scene.write_data_to_sim();env.sim.forward()
    env.mode='kneel';env.idle_target.copy_(env.robot.data.joint_pos[:,env.pids])
with torch.inference_mode():
    step=0
    while app.is_running() and (not args.steps or step<args.steps):
        started=time.monotonic()
        if state.consume_reset():env.request('reset')
        name=state.consume_pose() or (keyboard.pop_pose(env.mode) if keyboard else None) or schedule.get(step)
        if name:
            state.apply_joint_msg({'release_all':True})
            env.manual_overrides={}
            if name not in ['skate','slide']:env.command.zero_()
            accepted=env.request(name);requests.append({'step':step,'name':name,'accepted':accepted,'message':env.message})
            print('SKILL_REQUEST',json.dumps(requests[-1]),flush=True)
        with state._lock:
            command=state.web_cmd;state.web_cmd=None
        if command:apply_drive(env,command)
        if keyboard:
            keyboard_command=keyboard.command_update()
            if keyboard_command is not None:apply_drive(env,keyboard_command)
        env.manual_overrides=state.pop_overrides() if env.mode=='stand' else {}
        env.step(env.actor_action());env.after_step()
        snapshot=env.snapshot();state.publish(snapshot)
        if writer:
            frame=Image.fromarray(env.render());draw=ImageDraw.Draw(frame)
            draw.rectangle((0,0,960,52),fill=(0,0,0))
            draw.text((12,8),'Q1 | trained PPO skills | continuous physical simulation',fill='white')
            detail=f'arms error {snapshot["backflip"]["arms_error_rad"]:.3f} rad | flip hold {snapshot["backflip"]["hold_s"]:.2f}s' if env.mode=='backflip' else snapshot['message']
            draw.text((12,30),f'{step*env.step_dt:.2f}s | {env.mode} | {detail}',fill='white')
            writer.append_data(np.asarray(frame))
            if step in [0,300,400,650,900]:frame.save(Path(args.video_path).parent/f'frame_{step:04d}.png')
        if args.out and step%10==0:
            records.append({'step':step,**snapshot})
        step+=1
        if args.real_time:
            left=env.step_dt-(time.monotonic()-started)
            if left>0:time.sleep(left)
if args.out:
    out=Path(args.out);out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps({'requests':requests,'final':env.snapshot(),'records':records},indent=2))
if writer:writer.close()
if keyboard:keyboard.close()
env.close();app.close()
