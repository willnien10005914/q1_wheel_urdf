"""Collect real, legal PPO landing states for kneel-to-stand transfer resets."""
import argparse,json,sys,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];sys.path[:0]=[str(ROOT),str(ROOT/'q1_backflip_v2')]
from isaaclab.app import AppLauncher
p=argparse.ArgumentParser();p.add_argument('--num_envs',type=int,default=64);p.add_argument('--out',required=True)
AppLauncher.add_app_launcher_args(p);args=p.parse_args();app=AppLauncher(args).app
import torch
from skills.interactive_env import Env,Config
cfg=Config();cfg.scene.num_envs=args.num_envs;cfg.seed=4107;cfg.sim.device=args.device or 'cuda:0'
env=Env(cfg,ROOT);env.seed(4107);env.reset();env._clear_flip();env.mode='backflip'
bank=[]
with torch.inference_mode():
    for step in range(500):
        env.step(env.actor_action());env._state()
        env.hold_phase=torch.where(env.success&(env.hold_phase==0),env._phase(),env.hold_phase)
        if step>=250 and step%25==0:
            r=env.robot.data;error,speed=env.arm_status()
            valid=env.success&(~env.failed)&(error<.18)&(r.root_ang_vel_b.norm(dim=-1)<1.)&(r.root_lin_vel_w[:,:2].norm(dim=-1)<.35)
            for i in valid.nonzero().flatten().tolist():
                pos=r.root_pos_w[i]-env.scene.env_origins[i]
                bank.append({'env':i,'step':step,'root_pose':[0.,0.,float(pos[2])]+r.root_quat_w[i].tolist(),
                    'root_velocity':r.root_vel_w[i].tolist(),'joint_position':r.joint_pos[i].tolist(),
                    'joint_velocity':r.joint_vel[i].tolist(),'hold_s':float(env.best_hold[i]),'arm_error_rad':float(error[i])})
if len(bank)<16:raise RuntimeError(f'Insufficient real successful landings: {len(bank)}')
out=Path(args.out);out.parent.mkdir(parents=True,exist_ok=True)
out.write_text(json.dumps({'checkpoint_sha256':hashlib.sha256(Path(env.backflip_checkpoint).read_bytes()).hexdigest(),
    'joint_names':env.robot.joint_names,'states':bank,'physical_landings_only':True},indent=2))
print('LANDING_BANK',len(bank),flush=True);env.close();app.close()
