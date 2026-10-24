"""Continuous crouch -> rise -> wheel glide -> flip; no hidden reset after failure."""
import argparse,json,sys,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from isaaclab.app import AppLauncher
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--models',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--allow-preview',action='store_true');AppLauncher.add_app_launcher_args(p);a=p.parse_args();a.headless=True;a.enable_cameras=True;app=AppLauncher(a).app
import gymnasium as gym
import torch,numpy as np
import imageio.v2 as imageio
from PIL import Image,ImageDraw
from isaaclab_tasks.utils import parse_env_cfg
import wheel_humanoid_lab.tasks
from skills.policy import FrozenActor
from skills.reference_web_runtime import CONTRACT,read_json

env=None;results={};a.out.mkdir(parents=True,exist_ok=True)
try:
 cfg=parse_env_cfg('Isaac-Q1-Reference-Web-v0',device=a.device or 'cuda:0',num_envs=1);cfg.seed=9127;cfg.viewer.resolution=(960,640);cfg.sim.render_interval=8
 env=gym.make('Isaac-Q1-Reference-Web-v0',cfg=cfg,render_mode='rgb_array').unwrapped;s=env.command_manager.get_term('skill');actors={};metadata={}
 for skill in ('standup','kneel','walk','backflip','skate'):
  folder=a.models/skill;path=folder/'model.pt' if (folder/'model.pt').exists() else folder/'latest.pt';meta=read_json(folder/('validated.json' if path.name=='model.pt' else 'latest.json'))
  if not path.exists() or (path.name=='latest.pt' and not a.allow_preview):continue
  digest=hashlib.sha256(path.read_bytes()).hexdigest()
  if any(meta.get(k)!=v for k,v in dict(contract=CONTRACT,skill=skill,sha256=digest,reference_sha256=s.trajectory.sha256,urdf_sha256=s.trajectory.urdf_sha256).items()):continue
  actor=FrozenActor(path,env.device)
  if actor.input_dim!=90 or actor.mlp[-1].out_features!=23:raise ValueError('Recording policy shape mismatch')
  actors[skill]=actor;metadata[skill]=meta
 for name,total in [('crouch_rise_glide_backflip',1800),('alternating_skating',1400)]:
  env.reset(seed=9127);s.level=1.;s.hold_pose('kneel');current='kneel';first_failure=None;trace=[];selected={'kneel'};requests=[]
  writer=imageio.get_writer(str(a.out/(name+'.mp4')),fps=25,codec='libx264',quality=7,macro_block_size=16)
  try:
   with torch.inference_mode():
    for step in range(total):
     mode=None;vx=0.
     if name=='crouch_rise_glide_backflip':
      mode={100:'standup',750:'walk',1100:'stand_hold',1250:'backflip'}.get(step)
      vx=.3 if 750<=step<1000 else 0.
     else:
      mode={50:'standup',700:'slide'}.get(step);vx=.3 if 700<=step<1250 else 0.
     if mode:
      before=env.scene['robot'].data.root_state_w.clone()
      if mode=='stand_hold':s.hold_pose('stand');current='standup'
      else:s.set_mode(mode);current='skate' if mode=='slide' else mode
      if not torch.equal(before,env.scene['robot'].data.root_state_w):raise RuntimeError('Recording handover teleported robot')
      requests.append(dict(step=step,mode=mode,has_policy=current in actors));selected.add(current)
     cmd=env.command_manager.get_term('base_velocity');cmd.command[:]=torch.tensor([vx,0.,0.],device=env.device);cmd.time_left.fill_(1e9);cmd.is_standing_env.fill_(vx==0);s._update_command()
     obs=env.observation_manager.compute()['policy'];action=actors[current](obs) if current in actors else torch.zeros(1,23,device=env.device)
     env.step(action);r=env.scene['robot'].data;z=float(r.root_pos_w[0,2]-env.scene.env_origins[0,2]);up=float(-r.projected_gravity_b[0,2])
     failed=bool(s.illegal[0]) or z<.14 or (current!='backflip' and up<.45)
     if failed and first_failure is None:first_failure=step
     preview=not metadata.get(current,{}).get('validated',False)
     if step%25==0:trace.append(dict(step=step,skill=current,preview=preview,height=z,up=up,vx=float(r.root_lin_vel_b[0,0]),first_failure_step=first_failure,success=bool(s.success[0]),cycles=s.cycles.cycles[0].tolist()))
     if step%2==0:
      position=r.root_pos_w[0].cpu().numpy();target=np.array([position[0],position[1],max(.45,position[2])]);env.sim.set_camera_view(target+np.array([2.7,-3.,1.4]),target)
      frame=Image.fromarray(env.render());draw=ImageDraw.Draw(frame);draw.rectangle((0,0,960,84),fill='black')
      draw.text((10,6),'NEW Q1 | MediaPipe 15-23s | '+name,fill='white')
      draw.text((10,25),f'{step*.02:.2f}s | {current} | '+('UNVALIDATED PREVIEW / NOT SUCCESS MODEL' if preview else 'VALIDATED POLICY / continuous trial'),fill='yellow')
      draw.text((10,45),f'FAILED at {first_failure*.02:.2f}s / no resets' if first_failure is not None else 'FIRST TRIAL / no hidden resets',fill='red' if first_failure is not None else 'white')
      draw.text((10,64),f'z={z:.3f}m up={up:.3f} vx={float(r.root_lin_vel_b[0,0]):+.2f}',fill='white');writer.append_data(np.asarray(frame))
      if step in (0,total//2,total-2):frame.save(a.out/(name+f'_{step:04d}.png'))
  finally:writer.close()
  results[name]=dict(no_hidden_resets=True,first_failure_step=first_failure,requests=requests,trace=trace,models={k:metadata.get(k) for k in selected},missing_policies=sorted(selected-set(actors)),all_selected_models_validated=all(metadata.get(k,{}).get('validated') for k in selected),success_claimed=False)
  (a.out/'review.json').write_text(json.dumps(results,indent=2)+'\n')
finally:
 if env is not None:env.close()
 app.close()
