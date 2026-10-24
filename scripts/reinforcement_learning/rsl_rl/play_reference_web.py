"""Video-reference wheel Q1 player: boot at crouch, rise only on command."""
import argparse,json,math,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
from isaaclab.app import AppLauncher
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--web-port',type=int,default=8875);p.add_argument('--web-host',default='127.0.0.1')
p.add_argument('--no-browser',action='store_true');p.add_argument('--no-keyboard',action='store_true')
p.add_argument('--real-time',action='store_true');p.add_argument('--steps',type=int,default=0)
p.add_argument('--out',type=Path);p.add_argument('--sequence',default='')
p.add_argument('--campaign-check',action='store_true',help='Exercise all new motion commands in one continuous physical Web trial')
p.add_argument('--models',type=Path,default=ROOT/'checkpoints/reference_campaign_20261024')
p.add_argument('--training-state',type=Path,default=ROOT/'outputs/reference_campaign_20261024/status.json')
p.add_argument('--allow-preview',action='store_true',help='Explicit experimental playback of unvalidated checkpoints')
AppLauncher.add_app_launcher_args(p);a=p.parse_args();app=AppLauncher(a).app
import gymnasium as gym
import torch
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab_tasks.utils import parse_env_cfg
import wheel_humanoid_lab.tasks
from skills.reference_web_runtime import ReferenceWeb
from skills.teleop import IsaacKeyboard
from play_web import start_play_web

env=None;keyboard=None
try:
 cfg=parse_env_cfg('Isaac-Q1-Reference-Web-v0',device=a.device or 'cuda:0',num_envs=1)
 env=gym.make('Isaac-Q1-Reference-Web-v0',cfg=cfg).unwrapped
 if not isinstance(env,ManagerBasedRLEnv):raise TypeError('Web requires ManagerBasedRLEnv')
 env.reset(seed=42);runtime=ReferenceWeb(env,a.models,a.training_state,a.allow_preview)
 state=start_play_web(port=a.web_port,open_browser=not a.no_browser and not a.headless,host=a.web_host)
 if not a.headless and not a.no_keyboard:keyboard=IsaacKeyboard()
 schedule={int(k):v for k,v in (item.split(':',1) for item in a.sequence.split(',') if item)}
 if a.campaign_check:
  if a.sequence:raise ValueError('Campaign Web check owns its motion schedule')
  descent_end=20.+float(runtime.skill.kneel_duration)+4.
  events=[(.02,'unbox'),(14.,'skate'),(17.,'standard'),(20.,'kneel'),
          (descent_end,'unbox'),(descent_end+14.,'backflip'),(descent_end+24.,'standard'),
          (descent_end+27.,'slide'),(descent_end+36.,'standard')]
  schedule={round(at/env.step_dt):mode for at,mode in events};a.steps=round((descent_end+39.)/env.step_dt)
 step=0;requests=[];records=[];first_failure_step=None
 with torch.inference_mode():
  while app.is_running() and (not a.steps or step<a.steps):
   started=time.monotonic()
   if state.consume_reset():runtime.request('reset')
   name=state.consume_pose() or (keyboard.pop_pose(runtime.mode) if keyboard else None) or schedule.get(step)
   if name:
    state.apply_joint_msg({'release_all':True});accepted=runtime.request(name)
    if accepted:
     with state._lock:state.web_cmd=None
    requests.append(dict(step=step,name=name,accepted=accepted));print('REFERENCE_WEB_REQUEST',json.dumps(requests[-1]),flush=True)
   with state._lock:command=state.web_cmd;state.web_cmd=None
   if command is not None:runtime.drive(command)
   if keyboard:
    command=keyboard.command_update()
    if command is not None:runtime.drive(command)
   if a.campaign_check and runtime.mode in ('walk','slide'):
    elapsed=step*env.step_dt
    moving=14.<=elapsed<16.5 or descent_end+27.<=elapsed<descent_end+34.
    runtime.drive({'vx':.3 if moving else 0.})
   state.pop_overrides();runtime.motors.manual={}
   env.step(runtime.action());snapshot=runtime.snapshot();state.publish(snapshot)
   if a.campaign_check:
    r=env.scene['robot'].data
    finite=bool(torch.isfinite(r.root_state_w).all() and torch.isfinite(r.joint_pos).all())
    failed=not finite or runtime.failed or float(snapshot['base']['z'])<.14 or (runtime.mode!='backflip' and float(-r.projected_gravity_b[0,2])<.45)
    if failed and first_failure_step is None:first_failure_step=step
   if a.out and step%25==0:records.append(dict(step=step,pose=snapshot['pose'],base=snapshot['base'],phase=snapshot['motion_phase'],message=snapshot['message']))
   step+=1
   if a.real_time:time.sleep(max(0.,env.step_dt-(time.monotonic()-started)))
 if a.steps and step!=a.steps:raise RuntimeError('Player exited before finite step budget')
 if a.out:
  result=dict(steps=step,requests=requests,records=records,final=runtime.snapshot())
  if a.campaign_check:
   result['validation']=dict(first_trial_only=True,first_failure_step=first_failure_step,
    all_requested_commands_accepted=len(requests)==len(schedule) and all(item['accepted'] for item in requests),
    pass_validation=first_failure_step is None and len(requests)==len(schedule) and all(item['accepted'] for item in requests))
  a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(result,indent=2)+'\n')
finally:
 if keyboard:keyboard.close()
 if env is not None:env.close()
 app.close()
