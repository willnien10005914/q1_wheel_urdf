"""Collect button-ready physical standing states, without pose writes after Reset."""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];sys.path[:0]=[str(ROOT),str(ROOT/'q1_backflip_v2')]
from isaaclab.app import AppLauncher
p=argparse.ArgumentParser();p.add_argument('--num_envs',type=int,default=64);p.add_argument('--out',required=True)
AppLauncher.add_app_launcher_args(p);args=p.parse_args();app=AppLauncher(args).app
import torch
from skills.interactive_env import Env,Config
cfg=Config();cfg.scene.num_envs=args.num_envs;cfg.seed=4107;cfg.sim.device=args.device or 'cuda:0'
env=Env(cfg,ROOT);env.seed(4107);env.reset();bank=[]
with torch.inference_mode():
 for step in range(151):
  env.step(env.actor_action());r=env.robot.data
  if step>=50 and step%25==0:
   valid=(-r.projected_gravity_b[:,2]>.94)&((r.root_pos_w-env.scene.env_origins)[:,2]>.8)&(r.root_lin_vel_w[:,:2].norm(dim=-1)<.3)&((r.joint_pos[:,env.pids]-env.home).abs().amax(-1)<.15)
   for i in valid.nonzero().flatten().tolist():
    z=float((r.root_pos_w-env.scene.env_origins)[i,2]);bank.append({'env':i,'step':step,'root_pose':[0.,0.,z]+r.root_quat_w[i].tolist(),'root_velocity':r.root_vel_w[i].tolist(),'joint_position':r.joint_pos[i].tolist(),'joint_velocity':r.joint_vel[i].tolist()})
if len(bank)<16:raise RuntimeError('Not enough stable standing states: '+str(len(bank)))
out=Path(args.out);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps({'joint_names':env.robot.joint_names,'states':bank,'physical_standing_only':True},indent=2));print('STANDING_BANK',len(bank),flush=True)
env.close();app.close()
