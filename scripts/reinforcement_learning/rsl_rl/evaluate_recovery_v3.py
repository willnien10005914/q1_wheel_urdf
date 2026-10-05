"""Motor-only/learned contact controller evaluation; no root writes except env resets.
Default emits telemetry only. Cursor can enable --video and run each mode separately.
"""
import argparse,json,os,hashlib
from pathlib import Path
from isaaclab.app import AppLauncher
p=argparse.ArgumentParser();p.add_argument('--checkpoint');p.add_argument('--out',required=True);p.add_argument('--mode',choices=['both','supine','prone','boot_kneel'],default='both');p.add_argument('--steps',type=int,default=1499);p.add_argument('--video',action='store_true');p.add_argument('--num_envs',type=int,default=0);p.add_argument('--hold-kneel',action='store_true');p.add_argument('--record-stride',type=int,default=5);p.add_argument('--seed',type=int,default=42);p.add_argument('--stop-after-stand-s',type=float,default=0.0,help='If >0, stop once kneel_then_stand and stand_hold >= this many seconds');p.add_argument('--tail-after-stand-s',type=float,default=1.5,help='Extra seconds to keep recording after stand success before stopping');p.add_argument('--video-env-index',type=int,default=0,help='Camera / stop-gate env index (use with --num_envs to film a standing parallel env)');p.add_argument('--auto-video-stand-env',action='store_true',help='If set with --video and --num_envs>1, retarget camera to the first env that reaches stand success')
AppLauncher.add_app_launcher_args(p);a=p.parse_args();a.enable_cameras=a.video;app=AppLauncher(a).app
import torch,gymnasium as gym
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from rsl_rl.runners import OnPolicyRunner
import wheel_humanoid_lab.tasks
from wheel_humanoid_lab.tasks.manager_based.recovery_v3.env_cfg import (
 Q1RecoveryV3PlayCfg,Q1RecoveryV3SupinePlayCfg,Q1RecoveryV3PronePlayCfg,Q1RecoveryV3BootKneelPlayCfg,
)
from wheel_humanoid_lab.tasks.manager_based.recovery_v3.runner_cfg import Q1RecoveryV3PPORunnerCfg
from wheel_humanoid_lab.tasks.manager_based.recovery_v3.mdp import state
if a.video and a.mode=='both':raise ValueError('Record each mode separately: --mode supine|prone|boot_kneel')
if a.video and a.num_envs>1 and not (a.auto_video_stand_env or a.video_env_index>0):
 raise ValueError('Multi-env video needs --video-env-index N or --auto-video-stand-env')
O=Path(a.out);O.mkdir(parents=True,exist_ok=True)
if a.mode=='supine':
 cfg=Q1RecoveryV3SupinePlayCfg(); task_id='Isaac-Q1-RecoveryV3-Supine-Play-v0'
elif a.mode=='prone':
 cfg=Q1RecoveryV3PronePlayCfg(); task_id='Isaac-Q1-RecoveryV3-Prone-Play-v0'
elif a.mode=='boot_kneel':
 cfg=Q1RecoveryV3BootKneelPlayCfg(); task_id='Isaac-Q1-RecoveryV3-BootKneel-Play-v0'
else:
 cfg=Q1RecoveryV3PlayCfg(); task_id='Isaac-Q1-RecoveryV3-Play-v0'
cfg.seed=a.seed;cfg.scene.num_envs=a.num_envs or (2 if a.mode=='both' else 1)
# Wide spacing for video so neighbors stay out of frame (single-robot look).
cfg.scene.env_spacing=40. if a.video else 5.
if a.mode=='boot_kneel':
 cfg.events.reset_reference.params={'mode':0,'boot_kneel':True}
else:
 cfg.events.reset_reference.params['mode']=-1 if a.mode=='both' else ['supine','prone'].index(a.mode)
cfg.viewer.origin_type='env';cfg.viewer.env_index=int(a.video_env_index);cfg.viewer.eye=(2.8,-3.2,2.);cfg.viewer.lookat=(0,0,.5)
env=gym.make(task_id,cfg=cfg,render_mode='rgb_array' if a.video else None)
if a.video:env=gym.wrappers.RecordVideo(env,video_folder=str(O),name_prefix='recovery_eval_'+a.mode,step_trigger=lambda step:step==0,video_length=a.steps,disable_logger=True)
env=RslRlVecEnvWrapper(env);policy=None
if a.checkpoint:
 from importlib.metadata import version as _pkg_version
 from isaaclab_rl.rsl_rl import handle_deprecated_rsl_rl_cfg
 agent_cfg=Q1RecoveryV3PPORunnerCfg()
 # Split checkpoints: use matching runner experiment name for clarity.
 if a.mode=='supine':
  from wheel_humanoid_lab.tasks.manager_based.recovery_v3.runner_cfg import Q1RecoveryV3SupinePPORunnerCfg as _RC
  agent_cfg=_RC()
 elif a.mode=='prone':
  from wheel_humanoid_lab.tasks.manager_based.recovery_v3.runner_cfg import Q1RecoveryV3PronePPORunnerCfg as _RC
  agent_cfg=_RC()
 elif a.mode=='boot_kneel':
  from wheel_humanoid_lab.tasks.manager_based.recovery_v3.runner_cfg import Q1RecoveryV3BootKneelPPORunnerCfg as _RC
  agent_cfg=_RC()
 ver=_pkg_version('rsl-rl-lib')
 handle_deprecated_rsl_rl_cfg(agent_cfg, ver)
 import sys
 from pathlib import Path as _P
 sys.path.insert(0, str(_P(__file__).resolve().parents[3]/'tools'))
 from convert_rslrl5_checkpoint import convert as _convert_ckpt
 ckpt=str(_convert_ckpt(_P(a.checkpoint)))
 runner=OnPolicyRunner(env,agent_cfg.to_dict(),log_dir=None,device=str(env.unwrapped.device));runner.load(ckpt);policy=runner.get_inference_policy(device=env.unwrapped.device)
s=state(env.unwrapped);s.cap=3 if a.hold_kneel else 4 # Holding kneel diagnoses transfer separately from standing.
root=Path(__file__).resolve().parents[3]
source_paths=list((root/'source/wheel_humanoid_lab/wheel_humanoid_lab/tasks/manager_based/recovery_v3').glob('*.py'))+[root/'docs/reference/getup_candidates_v3'/name for name in ['motor_priors.json','supine.json','prone.json']]
source_sha256={str(path.relative_to(root)):hashlib.sha256(path.read_bytes()).hexdigest() for path in source_paths}
obs=env.get_observations();records=[];valid=torch.ones(env.num_envs,device=env.unwrapped.device,dtype=torch.bool);best_stand=s.stand_hold.clone();max_stage=torch.zeros(env.num_envs,device=env.unwrapped.device,dtype=torch.long);knelt=s.knelt.clone();stood=s.stood.clone();plant=s.planted.clone();best_hold=s.kneel_hold.clone();upright_knelt=s.upright_knelt.clone();upright_best=s.upright_hold.clone()
stop_hold=float(a.stop_after_stand_s);tail_steps=int(round(float(a.tail_after_stand_s)/env.unwrapped.step_dt)) if stop_hold>0 else 0
stand_ok_step=None;ran_steps=0;video_env=int(a.video_env_index)
with torch.inference_mode():
 for step in range(a.steps):
  if a.hold_kneel:s.cap=3
  action=policy(obs) if policy else torch.zeros((env.num_envs,env.num_actions),device=env.unwrapped.device)
  obs,_,done,_=env.step(action);s.update();valid&=~done.bool();best_stand=torch.maximum(best_stand,torch.where(valid,s.stand_hold,0.));max_stage=torch.maximum(max_stage,s.stage);knelt|=s.knelt&valid;stood|=s.stood&valid;plant|=s.planted&valid;best_hold=torch.maximum(best_hold,torch.where(valid,s.kneel_hold,0.));upright_knelt|=s.upright_knelt&valid;upright_best=torch.maximum(upright_best,torch.where(valid,s.upright_hold,0.))
  ran_steps=step+1
  if step%a.record_stride==0:
   q=s.robot.data.joint_pos;target=env.unwrapped.action_manager.get_term('joint_pos').processed_actions
   records.append({'step':step,'time_s':step*env.unwrapped.step_dt,'mode':s.mode.tolist(),'stage':s.stage.tolist(),'root_pos':(s.robot.data.root_pos_w-env.unwrapped.scene.env_origins).tolist(),'root_quat_wxyz':s.robot.data.root_quat_w.tolist(),'gravity':s.robot.data.projected_gravity_b.tolist(),'torso_upright':s.torso_upright.tolist(),'upright_phase':s.upright_active.tolist(),'upright_preposition':((s.upright_age < (s.upright_pre_duration+s.upright_pre_hold)[s.mode])&s.upright_active).tolist(),'upright_hold_s':s.upright_hold.tolist(),'assisted_kneel_hold_s':s.kneel_hold.tolist(),'stand_hold_s':s.stand_hold.tolist(),'stand_preload':((s.stage==4)&s.stand_enabled[s.mode]&(s.elapsed<s.stand_values[s.mode,11])).tolist(),'sit_back':s.stand_sit_back[s.mode].tolist(),'episode_valid':valid.tolist(),'wheel_velocity_targets':env.unwrapped.action_manager.get_term('wheel_vel').processed_actions.tolist(),'joint_pos':q.tolist(),'position_targets':target.tolist(),'arm_force_z':s.arm_force.tolist(),'support_force_z':s.support_force.tolist(),'wheel_clearance_approx':s.wheel_clearance.tolist(),'torque':s.robot.data.applied_torque.tolist(),'actuators':{name:{'current_a':act.current.tolist(),'saturated':act.saturated.tolist()} for name,act in s.robot.actuators.items() if hasattr(act,'current')},'knelt':s.knelt.tolist(),'stood':s.stood.tolist()})
  if step%500==0:print('physics',step,'stage',s.stage[:min(4,env.num_envs)].tolist(),'arm',s.arm_force[:min(4,env.num_envs)].tolist(),'wheel',s.support_force[:min(4,env.num_envs),:2].tolist(),flush=True)
  if stop_hold>0 and stand_ok_step is None:
   # stood flips at hold>=1.0s; allow earlier stop via --stop-after-stand-s when already stage-4 standing.
   ok=(best_stand>=stop_hold)&(s.stage>=4)&valid
   if a.auto_video_stand_env and bool(ok.any()):
    video_env=int(torch.where(ok)[0][0].item())
    cfg.viewer.env_index=video_env
    try:env.unwrapped.viewport_camera_controller.update_view_to_env()
    except Exception:pass
    print(f'[video] retarget camera to standing env_index={video_env}',flush=True)
   gate=video_env
   if float(best_stand[gate].item())>=stop_hold and int(s.stage[gate].item())>=4:
    stand_ok_step=step;print(f'[stop] stand success env={gate} at step={step} (~{step*env.unwrapped.step_dt:.1f}s) hold={float(best_stand[gate]):.2f}s stage={int(s.stage[gate])}; tail {tail_steps} steps',flush=True)
  if stand_ok_step is not None and step>=stand_ok_step+tail_steps:
   print(f'[stop] ending early at step={step}',flush=True);break
summary_extra={'ran_steps':ran_steps,'stopped_after_stand':stand_ok_step is not None,'stand_ok_step':stand_ok_step,'video_env_index':video_env,'stand_env_ids':torch.where(stood)[0].tolist()}
summary={'best_stand_hold_s':best_stand.tolist(),'first_episode_unbroken':valid.tolist(),'mode':s.mode.tolist(),'max_stage':max_stage.tolist(),'arm_plant':plant.tolist(),'kneel':knelt.tolist(),'kneel_then_stand':stood.tolist(),'best_kneel_hold_s':best_hold.tolist(),'upright_kneel':upright_knelt.tolist(),'best_upright_kneel_hold_s':upright_best.tolist(),**summary_extra}
(O/'evaluation.json').write_text(json.dumps({'source_sha256':source_sha256,'controller':'ppo' if policy else 'zero_action_motor_prior','checkpoint':a.checkpoint,'checkpoint_sha256':hashlib.sha256(Path(a.checkpoint).read_bytes()).hexdigest() if a.checkpoint else None,'seed':a.seed,'hold_kneel_only':a.hold_kneel,'summary':summary,'joint_names':s.robot.joint_names,'position_joint_names':[s.robot.joint_names[i] for i in s.pos_ids],'support_order':['left_wheel','right_wheel','left_roller','right_roller'],'records':records,'no_state_rewriting_between_resets':True},indent=2));print('RESULT',json.dumps(summary),flush=True)
env.close()
if a.video:
 source=O/('recovery_eval_'+a.mode+'-step-0.mp4')
 if source.exists():source.replace(O/('recovery_eval_'+a.mode+'.mp4'))
app.close()
