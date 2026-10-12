"""Physical regression for the live player. Run using Isaac Python with WHEEL_HUMANOID_ROOT set.

--headless --out result.json: sustained WASD, stops, then two full backflips.
--drive-only --num_envs 8: batch drive stability without synchronized skill handovers.
--turn-only: turn using the trained policy, stop, then backflip in the current heading.
"""
import argparse,sys,json,time
from pathlib import Path
root=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(root),str(root/'q1_backflip_v2')]
from isaaclab.app import AppLauncher
p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--num_envs',type=int,default=1);p.add_argument('--drive-only',action='store_true');p.add_argument('--turn-only',action='store_true');p.add_argument('--seed',type=int,default=4107);AppLauncher.add_app_launcher_args(p);args=p.parse_args();app=AppLauncher(args).app
import torch
from skills.interactive_env import Env,Config
from skills.teleop import apply_drive
cfg=Config();cfg.scene.num_envs=args.num_envs;cfg.seed=args.seed;cfg.sim.device=args.device or 'cuda:0'
if not args.drive_only and args.num_envs!=1:raise ValueError('Backflip UI validation uses one robot, matching the live player.')
env=Env(cfg,root);result={'seed':args.seed,'phases':[]}
def run(label,steps,cmd=None):
 samples=[]
 for step in range(steps):
  if cmd is not None:apply_drive(env,cmd)
  env.step(env.actor_action());env.after_step()
  if step%10==0:
   r=env.robot.data
   samples.append({'step':step,'mode':env.mode,'z':(r.root_pos_w[:,2]-env.scene.env_origins[:,2]).tolist(),'upright':(-r.projected_gravity_b[:,2]).tolist(),'speed':r.root_lin_vel_w[:,:2].norm(dim=-1).tolist(),'omega':r.root_ang_vel_b.norm(dim=-1).tolist(),'q_error':(r.joint_pos[:,env.pids]-env.home).abs().amax(-1).tolist(),'message':env.message,'winding':env.winding.tolist(),'max_progress':env.max_progress.tolist(),'pitch':r.projected_gravity_b[:,0].tolist(),'joint_speed':r.joint_vel[:,env.pids].abs().amax(-1).tolist(),'wheel_speed':r.joint_vel[:,env.wids].tolist(),'phase':env._phase().tolist()})
 phase={'label':label,'samples':samples,'final':env.snapshot()};result['phases'].append(phase)
 print('PHASE',label,'mode',env.mode,'z',samples[-1]['z'],'up',samples[-1]['upright'],'msg',env.message,flush=True)
env.request('reset');run('idle',150)
if args.turn_only:
 run('turn',175,{'yaw':.45});run('stop_turn',150,{})
 result['turn_backflip_accepted']=env.request('backflip');run('turn_backflip',600)
else:
 for label,steps,cmd in [('forward',150,{'vx':1.2}),('stop_forward',100,{}),('left',100,{'yaw':.45}),('stop_left',100,{}),('reverse',100,{'vx':-.55}),('stop_reverse',100,{}),('right',100,{'yaw':-.45}),('stop_right',150,{})]:run(label,steps,cmd)
 if not args.drive_only:
  result['drive_backflip_accepted']=env.request('backflip');run('drive_backflip',600)
  env.request('reset');run('reset_idle',200)
  result['idle_backflip_accepted']=env.request('backflip');run('idle_backflip',600)
output=Path(args.out);output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(result,indent=2));env.close();app.close()
assert all(min(min(r['z']) for r in phase['samples'])>.65 and min(min(r['upright']) for r in phase['samples'])>.85 for phase in result['phases'] if phase['label'] in ['forward','stop_forward','left','stop_left','reverse','stop_reverse','right','stop_right','turn','stop_turn']), 'Drive stability failed; inspect output JSON'
if not args.drive_only:
 assert all(phase['final']['backflip']['success'] for phase in result['phases'] if 'backflip' in phase['label']), 'Backflip did not finish the arms-down landing hold; inspect output JSON'
