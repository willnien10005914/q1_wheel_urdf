"""Independent one-robot trials of the actual web standing-to-backflip handover."""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];sys.path[:0]=[str(ROOT),str(ROOT/'q1_backflip_v2')]
from isaaclab.app import AppLauncher
p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--seeds',default='4107,5107,6107,7107,8107,9107,10107,11107');p.add_argument('--teacher')
AppLauncher.add_app_launcher_args(p);a=p.parse_args();app=AppLauncher(a).app
import torch
from skills.interactive_env import Env,Config
cfg=Config();cfg.scene.num_envs=1;cfg.seed=4107;cfg.sim.device=a.device or 'cuda:0';E=Env(cfg,ROOT,teacher_path=a.teacher)
results=[]
with torch.inference_mode():
 for seed in [int(x) for x in a.seeds.split(',')]:
  E.mode='stand';E.reset(seed=seed);E.idle_target.copy_(E.home);E.stand_anchor.copy_(E.robot.data.root_pos_w);E.ff_ready=False;E.pose_timer=0
  for step in range(100):E.step(E.actor_action());E.after_step()
  before_root=E.robot.data.root_state_w.clone();before_q=E.robot.data.joint_pos.clone()
  accepted=E.request('backflip');start_pose=E.robot.data.root_state_w[0].tolist();launched=None
  assert torch.equal(before_root,E.robot.data.root_state_w)
  assert torch.equal(before_q,E.robot.data.joint_pos)
  for step in range(750):
   E.step(E.actor_action());E.after_step()
   if E.mode=='backflip' and launched is None:launched=step
   if E.mode=='failed' or bool(E.success[0]):break
  error,speed=E.arm_status();row={'seed':seed,'accepted':accepted,'launch_wait_s':None if launched is None else launched*E.step_dt,'success':bool(E.success[0]) and not bool(E.failed[0]),'illegal':bool(E.failed[0]),'early_hop':bool(E.prehop[0]),'arm_error_rad':float(error[0]),'hold_s':float(E.best_hold[0]),'start_root_state':start_pose,'final':E.snapshot()};results.append(row)
  Path(a.out).write_text(json.dumps({'trials':len(results),'successes':sum(r['success'] for r in results),'first_trial_only':True,'actual_web_controller':True,'episodes':results},indent=2));print('WEB_BACKFLIP_TRIAL',json.dumps({k:v for k,v in row.items() if k not in ['final','start_root_state']}),flush=True)
E.close();app.close()
