"""Optimize the reference with the exact hot web controller and frozen trained PPO."""
import argparse,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];sys.path[:0]=[str(ROOT),str(ROOT/'q1_backflip_v2')]
from isaaclab.app import AppLauncher
p=argparse.ArgumentParser();p.add_argument('--num_envs',type=int,default=256);p.add_argument('--generations',type=int,default=8);p.add_argument('--replicas',type=int,default=8);p.add_argument('--out',required=True);p.add_argument('--initial',default='checkpoints/backflip_arm_rest/teacher.json');p.add_argument('--warmup',type=int,default=100)
AppLauncher.add_app_launcher_args(p);a=p.parse_args();app=AppLauncher(a).app
import torch
from skills.interactive_env import Env,Config
from arm_env import LOW,HIGH
cfg=Config();cfg.scene.num_envs=a.num_envs;cfg.seed=4107;cfg.sim.device=a.device or 'cuda:0'
E=Env(cfg,ROOT);E.batch_trials=True
lo=torch.tensor(LOW,device=E.device);hi=torch.tensor(HIGH,device=E.device)
mean=torch.tensor(json.loads(Path(a.initial).read_text())['parameters'],device=E.device);best_params=mean.clone();std=(hi-lo)*.04
N=E.num_envs;R=a.replicas;C=N//R;best=-1e9;history=[]
with torch.inference_mode():
 for generation in range(a.generations):
  candidates=(mean+std*torch.randn(C,len(mean),device=E.device)).clamp(lo,hi);candidates[0]=best_params;candidates[1]=mean
  E.mode='stand';E.reset(seed=4107+generation);E.idle_target.copy_(E.home);E.stand_anchor.copy_(E.robot.data.root_pos_w);E.ff_ready=False
  for step in range(a.warmup):E.step(E.actor_action());E.after_step()
  valid=(-E.robot.data.projected_gravity_b[:,2]>.94)&(E.robot.data.root_lin_vel_w[:,:2].norm(dim=-1)<.3)
  E.set_parameters(candidates.repeat_interleave(R,0));E._clear_flip();E.mode='backflip'
  for step in range(300):E.step(E.actor_action());E.after_step()
  good=E.success&~E.failed&~E.prehop&valid
  score=15*E.max_progress+40*(E.max_progress>5.9)+40*E.best_hold+300*good.float()-120*E.failed.float()-200*E.prehop.float()-200*(~valid).float()
  grouped=score.reshape(C,R);robust=grouped.mean(1)-.5*grouped.std(1,unbiased=False);order=robust.argsort(descending=True);i=int(order[0]);elites=candidates[order[:max(4,C//8)]]
  mean=.25*mean+.75*elites.mean(0);std=(.35*std+.65*elites.std(0)).clamp_min((hi-lo)*.005)
  if float(robust[i])>best:
   best=float(robust[i]);best_params=candidates[i].clone()
   Path(a.out).write_text(json.dumps({'parameters':best_params.tolist(),'environment':'arm_env','method':'CEM reference search on physical hot web handover; PPO frozen','warmup_steps':a.warmup,'robust_score':best,'generation':generation,'successful_replicas':int(good.reshape(C,R)[i].sum()),'replicas':R,'checkpoint':str(E.backflip_checkpoint)},indent=2))
  row={'generation':generation,'successful_trials':int(good.sum()),'valid_start':int(valid.sum()),'best_replicas':int(good.reshape(C,R)[i].sum()),'best_score':best,'max_turn':float(E.max_progress.max())};history.append(row);Path(a.out).with_suffix('.search.json').write_text(json.dumps(history,indent=2));print('HOT_SEARCH',json.dumps(row),flush=True)
E.close();app.close()
