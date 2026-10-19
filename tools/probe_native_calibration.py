"""Bounded motor-only ablations; no training, external support or hidden resets."""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from isaaclab.app import AppLauncher
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True);p.add_argument('--cases',nargs='*');AppLauncher.add_app_launcher_args(p);a=p.parse_args();a.headless=True;app=AppLauncher(a).app
import gymnasium as gym
import torch
from isaaclab_tasks.utils import parse_env_cfg
import wheel_humanoid_lab.tasks
from wheel_humanoid_lab.tasks.manager_based.inertia import native_skills as mdp
from skills.policy import FrozenActor
cases=[('kneel_controller','kneel',None,0.),('kneel_with_standup_ppo','kneel','standup',0.),
       ('standup_controller','standup',None,0.),('standup_latest','standup','standup',0.),
       ('walk_step_04','walk','walk',.4),('walk_step_12','walk','walk',1.2),
       ('backflip_controller','backflip',None,0.),('backflip_latest','backflip','backflip',0.)]
if a.cases:cases=[c for c in cases if c[0] in a.cases]
a.out.parent.mkdir(parents=True,exist_ok=True);results={};env=None
try:
 cfg=parse_env_cfg('Isaac-Q1-Native-Web-v0',device=a.device or 'cuda:0',num_envs=8);cfg.seed=9127
 env=gym.make('Isaac-Q1-Native-Web-v0',cfg=cfg).unwrapped;s=env.command_manager.get_term('skill');robot=env.scene['robot'];motor=env.action_manager.get_term('motors')
 stand=FrozenActor(ROOT/'checkpoints/native_deadline_20261019/stand/model.pt',env.device)
 actors={name:FrozenActor(ROOT/'checkpoints/native_deadline_20261019'/name/('model.pt' if name=='walk' else 'latest.pt'),env.device) for name in ('walk','standup','backflip')}
 feet,_=robot.find_bodies(['left_wheel_link','right_wheel_link'],preserve_order=True)
 legs=['left_hip_Pitch_joint','left_knee_1_Pitch_joint','right_hip_Pitch_joint','right_knee_1_Pitch_joint'];ids=[robot.joint_names.index(n) for n in legs];pids=[mdp.Q1_INERTIA_POSITION_JOINTS.index(n) for n in legs]
 with torch.inference_mode():
  for name,mode,actor_name,vx in cases:
   initial='standup' if mode=='standup' else 'stand';env.cfg.native_skill=initial;s.cfg.mode=initial;env.reset(seed=9127);s.level=1.
   failures=torch.zeros(8,dtype=torch.bool,device=env.device);failure_time=torch.full((8,),-1.,device=env.device);rows=[]
   start=0 if mode=='standup' else 100
   initial_q=robot.data.joint_pos[0,ids].tolist();initial_height=float(robot.data.root_pos_w[0,2]-env.scene.env_origins[0,2])
   for step in range(600):
    if step==start:s.set_mode(mode)
    command=env.command_manager.get_term('base_velocity');command.command[:]=torch.tensor([vx if step>=start else 0.,0.,0.],device=env.device);command.time_left.fill_(1e9);command.is_standing_env.fill_(not (vx and step>=start));s._update_command()
    obs=env.observation_manager.compute()['policy'];actor=stand if step<start else actors.get(actor_name)
    action=actor(obs) if actor else torch.zeros(8,23,device=env.device)
    if not torch.isfinite(action).all():raise FloatingPointError(name)
    env.step(action);r=robot.data;z=r.root_pos_w[:,2]-env.scene.env_origins[:,2];up=-r.projected_gravity_b[:,2]
    current_mode='stand' if step<start else mode
    failed=mdp.physical_failure(env) if current_mode!='kneel' else (z<.2)|(up<.5)|s.illegal
    new=failed&~failures;failure_time[new]=step*env.step_dt;failures|=failed
    if step%25==0:
     rows.append(dict(t=step*env.step_dt,mode=current_mode,z=float(z[0]),up=float(up[0]),vx=float(r.root_lin_vel_b[0,0]),q=r.joint_pos[0,ids].tolist(),reference=s.reference()[0,pids].tolist(),target=motor.target[0,pids].tolist(),raw_action=action[0,pids].tolist(),foot_bottom_z=(r.body_pos_w[0,feet,2]-env.scene.env_origins[0,2]-.1).tolist(),failed=bool(failures[0]),winding=float(s.winding[0])))
   results[name]={'trials':8,'first_failure_time_s':failure_time.tolist(),'failures':int(failures.sum()),'initial_leg_q':initial_q,'initial_root_z':initial_height,'leg_order':legs,'checkpoint_sha256':actors[actor_name].checkpoint_sha256 if actor_name else None,'trace':rows,'no_hidden_reset':True,'not_a_success_certification':True}
   a.out.write_text(json.dumps(results,indent=2)+'\n');print('CALIBRATION_PROBE',name,'failures',int(failures.sum()),'first',failure_time.tolist(),flush=True)
finally:
 if env is not None:env.close()
 app.close()
