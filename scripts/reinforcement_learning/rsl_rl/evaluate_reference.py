"""First-trial validation of video-reference policies; physical resets are masked."""
import argparse,json,sys,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
from isaaclab.app import AppLauncher
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--skill',choices=['standup','kneel','walk','backflip','skate'],required=True);p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--num_envs',type=int,default=64);p.add_argument('--steps',type=int,default=750);p.add_argument('--seed',type=int,default=9127);AppLauncher.add_app_launcher_args(p);a=p.parse_args();a.headless=True;app=AppLauncher(a).app
import gymnasium as gym
import torch
from isaaclab_tasks.utils import parse_env_cfg
import wheel_humanoid_lab.tasks
from skills.policy import FrozenActor
from wheel_humanoid_lab.tasks.manager_based.native_reference.trajectory import CONTRACT

env=None
try:
 task='Isaac-Q1-Reference-'+a.skill.capitalize()+'-v0';cfg=parse_env_cfg(task,device=a.device or 'cuda:0',num_envs=a.num_envs);cfg.seed=a.seed;cfg.curriculum=None;cfg.reference_phase_resets=False;cfg.observations.policy.enable_corruption=False
 env=gym.make(task,cfg=cfg).unwrapped;env.reset(seed=a.seed);s=env.command_manager.get_term('skill');s.level=1.
 actor=FrozenActor(a.checkpoint,env.device)
 if actor.input_dim!=90 or actor.mlp[-1].out_features!=23:raise ValueError('Reference policy shape mismatch')
 alive=torch.ones(a.num_envs,dtype=torch.bool,device=env.device);success=~alive;failed=~alive;illegal=~alive
 winding=torch.zeros(a.num_envs,device=env.device);hold=winding.clone();cycle=torch.zeros(a.num_envs,2,device=env.device)
 crouch_failed=torch.zeros_like(alive);crouch_contact=torch.ones_like(winding)
 crouch_max={name:winding.clone() for name in ('bilateral_loss_s','air_s','descent_mps','fast_descent_s','hop_s')}
 requested_steps=a.steps
 # The physical slow crouch must finish its retimed trajectory and hold before
 # validation ends; the legacy 750 steps only cover 15 seconds at 50 Hz.
 if a.skill=='kneel':a.steps=max(a.steps,int((s.kneel_duration+4.)/env.step_dt)+1)
 samples=upright=velocity_error=yaw_error=nonfinite=0.;trace=[]
 with torch.inference_mode():
  for step in range(a.steps):
   command=env.command_manager.get_term('base_velocity');vx=.3 if a.skill in ('walk','skate') and 100<=step<500 else 0.
   command.command[:]=torch.tensor([vx,0.,0.],device=env.device);command.time_left.fill_(1e9);command.is_standing_env.fill_(vx==0)
   s._update_command();obs=env.observation_manager.compute()['policy'];actions=actor(obs)
   if not torch.isfinite(actions).all():raise FloatingPointError('Nonfinite actor output')
   _,_,term,trunc,_=env.step(actions)
   last=getattr(env,'_reference_last',{});safe=alive&~term&~trunc
   success|=last.get('success',s.success)&alive&~term
   illegal|=last.get('illegal',s.illegal)&alive
   hold=torch.maximum(hold,torch.where(alive,last.get('best_hold',s.best_hold),0.))
   winding=torch.minimum(winding,torch.where(alive,last.get('winding',s.winding),0.))
   cycle=torch.maximum(cycle,torch.where(alive[:,None],last.get('cycles',s.cycles.cycles),0.))
   crouch_failed|=last.get('slow_kneel_failed',s.slow_kneel.failed)&alive
   contact=last.get('slow_kneel_contact_fraction',s.slow_kneel.contact_fraction)
   crouch_contact=torch.where(alive,contact,crouch_contact)
   for name in crouch_max:
    value=last.get('slow_kneel_max_'+name,getattr(s.slow_kneel,'max_'+name))
    crouch_max[name]=torch.maximum(crouch_max[name],torch.where(alive,value,0.))
   nonfinite+=int((env.termination_manager.get_term('nonfinite')&alive).sum())
   failed|=term&alive;r=env.scene['robot'].data
   samples+=int(safe.sum());upright+=int((safe&(-r.projected_gravity_b[:,2]>.9)).sum())
   velocity_error+=float(((r.root_lin_vel_b[:,0]-vx).abs()*safe).sum());yaw_error+=float((r.root_ang_vel_b[:,2].abs()*safe).sum())
   alive&=~term&~trunc
   if step%50==0:trace.append(dict(step=step,active=int(alive.sum()),successes=int(success.sum()),root_height=r.root_pos_w[:,2].tolist()))
   if not alive.any():break
 frac=float(success.float().mean());failure_frac=float(failed.float().mean());speed_error=velocity_error/max(1,samples);up=upright/max(1,samples);yaw=yaw_error/max(1,samples)
 if a.skill in ('walk','skate'):passed=up>=.95 and failure_frac<=.1 and speed_error<.2 and yaw<.2 and nonfinite==0
 else:passed=frac>=.8 and failure_frac<=.1 and nonfinite==0
 if a.skill=='skate':success=(cycle>=3).all(-1)&~failed;frac=float(success.float().mean());passed=passed and frac>=.8
 if a.skill=='backflip':passed=passed and int(illegal.sum())==0
 if a.skill=='kneel':
  # A policy cannot pass by reaching the last pose after jumping, falling,
  # repeatedly unloading one wheel, or succeeding only after an auto-reset.
  success&=~crouch_failed&(crouch_contact>=.95)&~failed
  frac=float(success.float().mean())
  passed=frac>=.8 and failure_frac<=.1 and nonfinite==0 and int(crouch_failed.sum())==0
 report=dict(contract=CONTRACT,skill=a.skill,first_trial_only=True,checkpoint_sha256=actor.checkpoint_sha256,checkpoint_iteration=actor.training_iteration,
  reference_sha256=s.trajectory.sha256,urdf_sha256=s.trajectory.urdf_sha256,seed=a.seed,trials=a.num_envs,steps=step+1,successes=int(success.sum()),failures=int(failed.sum()),illegal=int(illegal.sum()),nonfinite=nonfinite,
  upright_fraction=up,velocity_error_mps=speed_error,yaw_error_rps=yaw,best_hold_s=hold.tolist(),max_backturn_rad=(-winding).tolist(),valid_cycles_each_foot=cycle.tolist(),
  requested_steps=requested_steps,reference_kneel_duration_s=s.kneel_duration,
  slow_kneel=dict(violations=int(crouch_failed.sum()),bilateral_wheel_contact_fraction=crouch_contact.tolist(),
   **{'max_'+name:value.tolist() for name,value in crouch_max.items()},required_contact_fraction=.95,settling_grace_s=.2,
   max_bilateral_loss_allowed_s=.12,max_air_allowed_s=.06,max_fast_descent_allowed_s=.1,
   fast_descent_threshold_mps=.38,max_hop_allowed_s=.08,reference_max_height_speed_mps=s.crouch_trajectory.max_height_speed,
   reference_max_joint_speed_rps=s.crouch_trajectory.max_joint_speed),
  pass_validation=bool(passed),score=frac+up-speed_error-failure_frac,trace=trace)
 a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');print('REFERENCE_EVALUATION',json.dumps({k:v for k,v in report.items() if k not in ('trace','best_hold_s','max_backturn_rad','valid_cycles_each_foot')}),flush=True)
finally:
 if env is not None:env.close()
 app.close()
