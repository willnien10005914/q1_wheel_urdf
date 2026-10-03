"""Contact-gated recovery, CubeMars dynamics. Simulation state writes: reset ONLY.
Candidate angles are exploration priors, not demonstrations or root tracking targets.
"""
import json
from pathlib import Path
import torch
from isaaclab.utils import configclass
from isaaclab.envs.mdp.actions.joint_actions import JointPositionAction, JointVelocityAction
from isaaclab.envs.mdp.actions.actions_cfg import JointPositionActionCfg, JointVelocityActionCfg
from wheel_humanoid_lab import WHEEL_HUMANOID_ROOT_DIR
from wheel_humanoid_lab.assets import Q1_POSITION_JOINTS

class State:
 def __init__(self,env):
  self.env=env;self.robot=env.scene['robot'];self.sensor=env.scene['contact_forces'];self.device=env.device;n=env.num_envs
  docs=[json.loads((Path(WHEEL_HUMANOID_ROOT_DIR)/'docs/reference/getup_candidates_v3'/f'{m}.json').read_text()) for m in ['supine','prone']]
  names=self.robot.joint_names
  self.initial_q=torch.tensor([[d['trajectory'][0]['joints'].get(k,0) for k in names] for d in docs],device=self.device)
  self.initial_root=torch.tensor([d['trajectory'][0]['root']['position']+d['trajectory'][0]['root']['quaternion_wxyz'] for d in docs],device=self.device)
  self.targets=torch.tensor([[[d['keyframes'][i]['joints'].get(k,0) for k in names] for i in [1,2,3,4,6]] for d in docs],device=self.device)
  prior=json.loads((Path(WHEEL_HUMANOID_ROOT_DIR)/'docs/reference/getup_candidates_v3/motor_priors.json').read_text())
  self.targets=torch.tensor([[[q.get(k,0.) for k in names] for q in prior['targets'][m]] for m in ['supine','prone']],device=self.device)
  self.duration=torch.tensor([prior.get('transition_duration_s',{}).get(m,[2.]*5) for m in ['supine','prone']],device=self.device)
  self.capture_after=float(prior.get('capture_kneel_after_s',.12))
  self.stand_after=float(prior.get('stand_after_kneel_hold_s',2.))
  self.stand_enabled=torch.tensor([prior.get('standing_transfer',{}).get(m,{}).get('enabled',False) for m in ['supine','prone']],device=self.device)
  self.stand_allowed=self.stand_enabled.clone()
  self.stand_sit_back=torch.tensor([prior.get('standing_transfer',{}).get(m,{}).get('sit_back',False) for m in ['supine','prone']],device=self.device)
  self.stand_values=torch.tensor([(prior.get('standing_transfer',{}).get(m,{}).get('parameters',[0.]*15)+[0.,0.])[:15] for m in ['supine','prone']],device=self.device)
  self.transfer_held=torch.zeros(n,dtype=torch.bool,device=self.device)
  self.held_command=torch.zeros((n,len(names)),device=self.device)
  self.torso_body_id=self.robot.find_bodies('torso')[0][0]
  self.upright_target=self.targets[:,3].clone();self.upright_duration=torch.ones(2,device=self.device)*4
  self.upright_pre_target=self.upright_target.clone();self.upright_pre_duration=torch.zeros(2,device=self.device);self.upright_pre_hold=torch.zeros(2,device=self.device)
  self.upright_enabled=torch.zeros(2,dtype=torch.bool,device=self.device)
  for mode,name in enumerate(['supine','prone']):
   spec=prior.get('upright_transfer',{}).get(name,{})
   if spec.get('enabled',False):
    self.upright_enabled[mode]=True;values=spec['parameters'];self.upright_duration[mode]=values[5]
    for joint,value in zip(['hip_pitch','knee','shoulder_pitch','elbow','waist_pitch'],values[:5]):
     for n_joint in ([joint+'_joint'] if joint.startswith('waist') else [f'{side}_{joint}_joint' for side in ['l','r']]):self.upright_target[mode,names.index(n_joint)]=value
   prep=spec.get('preposition_parameters')
   if prep:
    self.upright_pre_duration[mode]=prep[5];self.upright_pre_hold[mode]=spec.get('preposition_hold_s',1.)
    for joint,value in zip(['hip_pitch','knee','shoulder_pitch','elbow','waist_pitch'],prep[:5]):
     for n_joint in ([joint+'_joint'] if joint.startswith('waist') else [f'{side}_{joint}_joint' for side in ['l','r']]):self.upright_pre_target[mode,names.index(n_joint)]=value
  self.upright_active=torch.zeros(n,dtype=torch.bool,device=self.device)
  self.upright_captured=self.upright_active.clone();self.upright_knelt=self.upright_active.clone()
  self.upright_age=torch.zeros(n,device=self.device);self.upright_hold=self.upright_age.clone()
  self.upright_start=self.held_command.clone();self.upright_command=self.held_command.clone()
  self.mode=torch.arange(n,device=self.device)%2;self.stage=torch.zeros(n,dtype=torch.long,device=self.device)
  self.pos_ids=self.robot.find_joints(Q1_POSITION_JOINTS,preserve_order=True)[0]
  self.shoulder_ids=self.robot.find_joints(['l_shoulder_pitch_joint','r_shoulder_pitch_joint'],preserve_order=True)[0]
  self.wheel_ids=self.robot.find_bodies(['l_wheel_link','r_wheel_link'],preserve_order=True)[0]
  # Elbows plant; wrist/gripper must not jam the floor (tracked separately for penalties).
  self.elbow_names=[f'{s}_elbow_link' for s in ['l','r']]
  self.wrist_names=[f'{s}_wrist_link' for s in ['l','r']]
  self.gripper_names=[f'{s}_gripper_link' for s in ['l','r']]
  self.forearm_names=self.elbow_names+self.wrist_names  # plant surface: elbow/forearm, not gripper tip
  self.distal_names=self.wrist_names+self.gripper_names
  self.hand_names=self.elbow_names+self.distal_names
  self.elbow_ids=self.sensor.find_bodies(self.elbow_names,preserve_order=True)[0]
  self.wrist_ids=self.sensor.find_bodies(self.wrist_names,preserve_order=True)[0]
  self.gripper_ids=self.sensor.find_bodies(self.gripper_names,preserve_order=True)[0]
  self.forearm_ids=self.sensor.find_bodies(self.forearm_names,preserve_order=True)[0]
  self.distal_ids=self.sensor.find_bodies(self.distal_names,preserve_order=True)[0]
  self.hand_ids=self.sensor.find_bodies(self.hand_names,preserve_order=True)[0]
  self.support_ids=self.sensor.find_bodies(['l_wheel_link','r_wheel_link','l_knee_roller_link','r_knee_roller_link'],preserve_order=True)[0]
  self.roller_body_ids=self.robot.find_bodies(['l_knee_roller_link','r_knee_roller_link'],preserve_order=True)[0]
  self.arm_joint_ids=self.robot.find_joints(['.*_shoulder_.*_joint','.*_elbow_joint'])[0]
  self.waist_joint_ids=self.robot.find_joints(['waist_yaw_joint','waist_roll_joint','waist_pitch_joint'],preserve_order=True)[0]
  self.hip_roll_ids=self.robot.find_joints(['l_hip_roll_joint','r_hip_roll_joint'],preserve_order=True)[0]
  self.hip_pitch_ids=self.robot.find_joints(['l_hip_pitch_joint','r_hip_pitch_joint'],preserve_order=True)[0]
  self.filtered_force=torch.zeros_like(self.sensor.data.net_forces_w[:,:,2])
  self.elbow_body_ids=self.robot.find_bodies(self.elbow_names,preserve_order=True)[0]
  self.forearm_body_ids=self.robot.find_bodies(self.forearm_names,preserve_order=True)[0]
  self.gripper_body_ids=self.robot.find_bodies(self.gripper_names,preserve_order=True)[0]
  self.distal_body_ids=self.robot.find_bodies(self.distal_names,preserve_order=True)[0]
  self.hand_body_ids=self.robot.find_bodies(self.hand_names,preserve_order=True)[0]
  self.elapsed=torch.zeros(n,device=self.device);self.hold=self.elapsed.clone();self.kneel_hold=self.elapsed.clone();self.stand_hold=self.elapsed.clone()
  self.knelt=torch.zeros(n,dtype=torch.bool,device=self.device);self.stood=self.knelt.clone();self.planted=self.knelt.clone()
  self.start_q=self.initial_q[self.mode].clone();self.command=self.start_q.clone();self.last_step=-1;self.dirty=False
  self.totals=torch.zeros((2,5),device=self.device);self.pulse=torch.zeros(n,device=self.device)
  self.cap=3 # Stand unlocked by BOTH modes' real floor-to-kneel successes, never RSI.
 def update(self):
  e=self.env
  fresh=self.last_step!=e.common_step_counter
  if not fresh and not self.dirty:return
  dt=e.step_dt if fresh else 0.
  self.last_step=e.common_step_counter;self.dirty=False
  self.elapsed+=dt
  if fresh:self.pulse.zero_()
  # Positive vertical force, not force magnitude: side collisions do not count as support.
  if fresh:self.filtered_force.lerp_(self.sensor.data.net_forces_w[:,:,2].clamp(min=0),.2)
  f=self.filtered_force
  self.support_force=f[:,self.support_ids]
  self.elbow_force=f[:,self.elbow_ids]
  self.wrist_force=f[:,self.wrist_ids]
  self.gripper_force=f[:,self.gripper_ids]
  # Forearm plant = elbow + wrist (physical contact often lands on forearm). Gripper tip excluded.
  self.forearm_force=self.elbow_force+self.wrist_force
  self.distal_force=self.wrist_force+self.gripper_force
  self.arm_force=self.forearm_force  # 118-D obs arm channel
  self.support=self.support_force>5.;self.arms=self.arm_force>5.;self.grippers=self.gripper_force>5.
  self.height=self.robot.data.root_pos_w[:,2]-e.scene.env_origins[:,2]
  self.upright=-self.robot.data.projected_gravity_b[:,2]
  quat=self.robot.data.body_quat_w[:,self.torso_body_id]
  self.torso_upright=1-2*(quat[:,1].square()+quat[:,2].square())
  self.face_down=self.robot.data.projected_gravity_b[:,0]>.5
  self.shoulder=self.robot.data.joint_pos[:,self.shoulder_ids]
  self.wheel_clearance=(self.robot.data.body_pos_w[:,self.wheel_ids,2]-e.scene.env_origins[:,None,2]-.1).clamp(min=0)
  self.elbow_height=(self.robot.data.body_pos_w[:,self.elbow_body_ids,2]-e.scene.env_origins[:,None,2])
  self.forearm_height=(self.robot.data.body_pos_w[:,self.forearm_body_ids,2]-e.scene.env_origins[:,None,2]).reshape(-1,2,2).min(-1).values
  self.gripper_height=(self.robot.data.body_pos_w[:,self.gripper_body_ids,2]-e.scene.env_origins[:,None,2])
  self.hand_height=self.forearm_height
  # Plant when forearms load the floor (elbow/wrist). Grippers may brush but do not count.
  arm_ready=self.arms.all(-1)&((self.mode==1)|(self.shoulder.min(-1).values>1.8))
  self.planted|=arm_ready
  # Feet wheels + knee rollers still define kneel/stand (same as last standing policy).
  kneel=self.support.all(-1)&(self.height>.32)&(self.height<.65)&(self.upright>.7)&(self.robot.data.root_lin_vel_w.norm(dim=-1)<.6)
  stand=self.support[:,:2].all(-1)&(self.height>.72)&(self.upright>.9)&(self.torso_upright>.9)&(self.arm_force.max(-1).values<40)&(self.gripper_force.max(-1).values<10)&(self.robot.data.root_lin_vel_w.norm(dim=-1)<.5)
  upright_kneel=kneel&(self.torso_upright>.85)&(self.arm_force.max(-1).values<40)&(self.gripper_force.max(-1).values<10)
  self.upright_hold=torch.where(upright_kneel,self.upright_hold+dt,0.)
  self.upright_knelt|=self.upright_hold>=.5
  self.kneel_hold=torch.where(kneel,self.kneel_hold+dt,0.)
  self.stand_hold=torch.where(stand,self.stand_hold+dt,0.)
  new_kneel=(self.kneel_hold>=.4)&(~self.knelt)
  self.knelt|=new_kneel;self.pulse+=new_kneel.float()*5
  self.stood|=(self.stand_hold>=1.)&self.knelt
  ready=torch.where(self.stage==0,arm_ready,torch.where(self.stage==1,self.support[:,:2].all(-1)&self.arms.any(-1),torch.where(self.stage==2,self.support.all(-1)&(self.height>.30),torch.where(self.stage==3,upright_kneel,stand))))
  self.hold=torch.where(ready,self.hold+dt,0.)
  required_hold=torch.where(self.stage==3,self.stand_after,.35)
  # Finish the planned motor-prior blend before advancing. The old elapsed>=1.5 cut
  # 2 s transitions at ~75% completion and read as a sudden fast-forward on video.
  min_elapsed=self.duration[self.mode,self.stage]
  advance=(self.hold>=required_hold)&(self.elapsed>=min_elapsed)&(self.stage<self.cap)
  advance&=(self.stage<3)|self.stand_allowed[self.mode]
  # Keep the first stage transition grounded. Never advance merely because a timer expired.
  self.start_q[advance]=self.command[advance];self.stage[advance]+=1;self.elapsed[advance]=0.;self.hold[advance]=0.;self.pulse+=advance.float()
  capture=(self.stage==3)&(self.kneel_hold>=self.capture_after)&(~self.transfer_held)
  self.held_command[capture]=self.command[capture];self.transfer_held|=capture
  u=(self.elapsed/self.duration[self.mode,self.stage]).clamp(0,1);u=u*u*(3-2*u)
  target=self.targets[self.mode,self.stage] if not hasattr(self,"search_targets") else self.search_targets[torch.arange(e.num_envs,device=self.device),self.stage]
  # A loaded arm must change angle as the torso pivots. Freezing the old shoulder
  # angle was preventing transfer. Stop the motor interpolation once physical
  # kneeling is reached; do not continue through the successful posture.
  self.command=torch.lerp(self.start_q,target,u[:,None])
  hold=(self.stage==3)&self.transfer_held
  self.command[hold]=self.held_command[hold]
  begin=(self.stage==3)&self.transfer_held&(self.kneel_hold>=.8)&self.upright_enabled[self.mode]&(~self.upright_active)
  self.upright_start[begin]=self.command[begin];self.upright_active|=begin
  running=self.upright_active&(self.stage==3)
  self.upright_age[running]+=dt
  capture_upright=running&(self.upright_hold>=.12)&(~self.upright_captured)
  # Save the command that produced the physical contact result, before updating its next target.
  self.upright_captured|=capture_upright
  v=(self.upright_age/self.upright_duration[self.mode]).clamp(0,1);v=v*v*(3-2*v)
  target_upright=torch.lerp(self.upright_start,self.upright_target[self.mode],v[:,None])
  prep_duration=self.upright_pre_duration[self.mode];prep_total=prep_duration+self.upright_pre_hold[self.mode]
  pre=(self.upright_age/prep_duration.clamp(min=.02)).clamp(0,1);pre=pre*pre*(3-2*pre)
  post=((self.upright_age-prep_total)/self.upright_duration[self.mode]).clamp(0,1);post=post*post*(3-2*post)
  staged=torch.where((self.upright_age<prep_total)[:,None],torch.lerp(self.upright_start,self.upright_pre_target[self.mode],pre[:,None]),torch.lerp(self.upright_pre_target[self.mode],self.upright_target[self.mode],post[:,None]))
  target_upright=torch.where((prep_duration>0)[:,None],staged,target_upright)
  moving=running&(~self.upright_captured)
  self.upright_command[moving]=target_upright[moving]
  self.command[running]=self.upright_command[running]
  standing=(self.stage==4)&self.stand_enabled[self.mode]
  if standing.any():
   values=self.stand_values[self.mode];age=self.elapsed+e.step_dt
   preload=self.start_q.clone();preload[:,self.robot.joint_names.index('waist_pitch_joint')]=values[:,9]
   for side in ['l','r']:preload[:,self.robot.joint_names.index(side+'_shoulder_pitch_joint')]=values[:,10]
   sit=self.stand_sit_back[self.mode]
   for side in ['l','r']:
    preload[sit,self.robot.joint_names.index(side+'_hip_pitch_joint')]=values[sit,13];preload[sit,self.robot.joint_names.index(side+'_knee_joint')]=values[sit,14]
   a=(age/values[:,11].clamp(min=.02)).clamp(0,1);a=a*a*(3-2*a)
   b=((age-values[:,11])/values[:,5].clamp(min=.02)).clamp(0,1);b=b*b*(3-2*b)
   command=torch.where((age<values[:,11])[:,None],torch.lerp(self.start_q,preload,a[:,None]),torch.lerp(preload,self.targets[self.mode,4],b[:,None]))
   self.command[standing]=command[standing]

def state(env):
 if not hasattr(env,'_recovery_v3'):env._recovery_v3=State(env)
 return env._recovery_v3

def reset(env,env_ids,mode=-1):
 s=state(env)
 if env_ids is None:env_ids=torch.arange(env.num_envs,device=env.device)
 s.mode[env_ids]=env_ids%2 if mode<0 else mode
 s.upright_active[env_ids]=False;s.upright_captured[env_ids]=False;s.upright_knelt[env_ids]=False;s.upright_age[env_ids]=0.;s.upright_hold[env_ids]=0.;s.upright_command[env_ids]=0.
 s.transfer_held[env_ids]=False;s.held_command[env_ids]=0.
 s.stage[env_ids]=0;s.elapsed[env_ids]=0;s.hold[env_ids]=0;s.kneel_hold[env_ids]=0;s.stand_hold[env_ids]=0;s.knelt[env_ids]=False;s.stood[env_ids]=False;s.planted[env_ids]=False
 q=s.initial_q[s.mode[env_ids]];root=s.initial_root[s.mode[env_ids]].clone();root[:,:3]+=env.scene.env_origins[env_ids];root[:,2]+=.02
 s.start_q[env_ids]=q;s.command[env_ids]=q
 s.robot.write_joint_state_to_sim(q,torch.zeros_like(q),env_ids=env_ids)
 s.robot.write_root_pose_to_sim(root,env_ids=env_ids)
 s.robot.write_root_velocity_to_sim(torch.zeros((len(env_ids),6),device=env.device),env_ids=env_ids)
 # Clear derived values through next observation update, including initial reset at step 0.
 s.filtered_force[env_ids]=0.;s.dirty=True

def observation(env):
 s=state(env);s.update()
 return torch.cat((torch.nn.functional.one_hot(s.mode,2),torch.nn.functional.one_hot(s.stage,5),s.command[:,s.pos_ids],(s.arm_force/100).clamp(0,2),(s.support_force/100).clamp(0,2),s.wheel_clearance,s.upright[:,None],s.height[:,None],(s.hold/.4).clamp(0,1)[:,None]),-1)

class ContactPositionAction(JointPositionAction):
 def process_actions(self,actions):
  self._raw_actions[:]=actions;s=state(self._env)
  # Full soft-limit range remains reachable. Zero action follows contact-stage priors.
  lim=self._asset.data.soft_joint_pos_limits[:,self._joint_ids];mid=lim.mean(-1);half=(lim[:,:,1]-lim[:,:,0])/2
  bias=torch.atanh(((s.command[:,self._joint_ids]-mid)/half).clamp(-.98,.98))
  # From kneel onward, keep arms near motor prior (reduces shake that also blocks
  # stand success which requires arm_force < 30). Early floor stages stay free.
  calm=s.stage>=3
  gain=torch.where(calm,.15,.65)[:,None]
  processed=mid+half*torch.tanh(gain*actions+bias)
  if not hasattr(self,'_arm_action_ids'):
   ids=self._joint_ids.tolist() if torch.is_tensor(self._joint_ids) else list(self._joint_ids)
   names=[self._asset.joint_names[int(i)] for i in ids]
   self._arm_action_ids=[i for i,n in enumerate(names) if ('shoulder' in n or 'elbow' in n)]
   self._distal_action_ids=[i for i,n in enumerate(names) if ('wrist' in n or 'gripper' in n)]
   self._hip_roll_action_ids=[i for i,n in enumerate(names) if 'hip_roll' in n]
   # waist + hip_pitch stay free so PPO can use high-torque hips and waist to rise
  if self._arm_action_ids:
   idx=torch.tensor(self._arm_action_ids,device=processed.device,dtype=torch.long)
   prior=s.command[:,self._joint_ids][:,idx]
   blend=torch.where(calm,.92,.0)[:,None]
   arm=processed[:,idx]
   processed=processed.clone();processed[:,idx]=blend*prior+(1-blend)*arm
  # Wrists/grippers stay on motor prior — elbows do the floor work.
  if self._distal_action_ids:
   idx=torch.tensor(self._distal_action_ids,device=processed.device,dtype=torch.long)
   prior=s.command[:,self._joint_ids][:,idx]
   processed=processed.clone();processed[:,idx]=.97*prior+.03*processed[:,idx]
  # Keep feet from pigeon-toeing (內八): motor priors hold hip_roll at 0; damp PPO residuals.
  # hip_pitch is intentionally NOT damped — AKE90 hips must drive the sit-up.
  if self._hip_roll_action_ids:
   idx=torch.tensor(self._hip_roll_action_ids,device=processed.device,dtype=torch.long)
   prior=s.command[:,self._joint_ids][:,idx]
   processed=processed.clone();processed[:,idx]=.75*prior+.25*processed[:,idx]
  self._processed_actions=processed
@configclass
class ContactPositionActionCfg(JointPositionActionCfg):
 class_type:type=ContactPositionAction
 use_default_offset:bool=False

def balance_velocity(s):
 values=s.stand_values[s.mode]
 pitch=torch.atan2(s.robot.data.projected_gravity_b[:,0],-s.robot.data.projected_gravity_b[:,2])
 target=values[:,6]*(pitch-values[:,8])+values[:,7]*s.robot.data.root_ang_vel_b[:,1]+values[:,12]*s.robot.data.root_lin_vel_b[:,0]
 # Keep foot wheels still during the four-contact sit-back; balance starts with leg extension.
 ready=(~s.stand_sit_back[s.mode])|(s.elapsed+s.env.step_dt>=values[:,11])
 return target*((s.stage==4)&s.stand_enabled[s.mode]&ready)

class ContactWheelVelocityAction(JointVelocityAction):
 def process_actions(self,actions):
  # PPO adds a bounded residual to a physical balance prior. CubeMars applies torque/current limits.
  super().process_actions(actions)
  s=state(self._env)
  # At kneel (stage 3) PPO often commanded ±8 rad/s ("轉圈"). Clamp hard there;
  # keep full residuals for floor plant (0–2) and stand balance (4).
  residual=self._processed_actions
  kneeling=s.stage==3
  residual=torch.where(kneeling[:,None],residual.clamp(-1.5,1.5),residual)
  self._processed_actions=(residual+balance_velocity(s).clamp(-25.,25.)[:,None]).clamp(-25.,25.)
@configclass
class ContactWheelVelocityActionCfg(JointVelocityActionCfg):
 class_type:type=ContactWheelVelocityAction

def reward(env,kind):
 s=state(env);s.update();early=(s.stage<=2).float();sup=(s.mode==0).float();prone=(s.mode==1).float()
 forearm=(s.forearm_force/40).clamp(0,1).min(-1).values
 grip=(s.gripper_force/20).clamp(0,1).max(-1).values
 wheels=(s.support_force[:,:2]/60).clamp(0,1).min(-1).values
 rollers=(s.support_force[:,2:]/60).clamp(0,1).min(-1).values
 if kind=='approach':return early*(torch.exp(-s.forearm_height.clamp(min=0)/.12).mean(-1)+(s.stage>=1).float()*torch.exp(-s.wheel_clearance/.12).mean(-1))
 # Forearm (elbow/wrist) plant; lightly discount gripper tip loading.
 if kind=='plant':return (s.stage==0).float()*forearm*(1.-.5*grip)*torch.where(s.mode==0,((s.shoulder.min(-1).values-1.5)/.5).clamp(0,1),1.)
 # Foot active wheels + knee passive rollers are first-class through kneel.
 if kind=='wheel':return (s.stage>=1).float()*wheels
 if kind=='kneel':return (s.stage>=2).float()*wheels*rollers*s.upright.clamp(0,1)*s.torso_upright.clamp(0,1).square()*torch.exp(-((s.height-.46)/.15).square())
 if kind=='progress':return s.pulse/env.step_dt
 if kind=='lift':
  roller_z=(s.robot.data.body_pos_w[:,s.roller_body_ids,2]-env.scene.env_origins[:,None,2]).clamp(min=0)
  return (s.stage>=2).float()*wheels*(2*s.upright.clamp(0,1)+2*s.torso_upright.clamp(0,1)+torch.exp(-((s.height-.46)/.18).square())+torch.exp(-roller_z/.18).min(-1).values)
 if kind=='stand':
  arm_speed=s.robot.data.joint_vel[:,s.arm_joint_ids].abs().mean(-1)
  calm=torch.exp(-arm_speed/.8)*torch.exp(-(s.forearm_force.max(-1).values.clamp(min=0)/40))*torch.exp(-(s.gripper_force.max(-1).values.clamp(min=0)/10))
  return s.knelt.float()*(s.stage==4).float()*wheels*s.upright.clamp(0,1)*torch.exp(-((s.height-.84)/.12).square())*torch.exp(-s.robot.data.root_lin_vel_w.square().sum(-1)/.25)*calm
 if kind=='arm_assist':
  rising=(s.stage<=2)|((s.stage==4)&(s.elapsed<(s.stand_values[s.mode,11]+s.stand_values[s.mode,5])))
  return rising.float()*forearm*(1.-.5*grip)*torch.exp(-s.forearm_height.clamp(min=0)/.18).mean(-1)
 if kind=='arm_calm':
  arm_speed=s.robot.data.joint_vel[:,s.arm_joint_ids].abs().mean(-1)
  residual=(s.robot.data.joint_pos[:,s.arm_joint_ids]-s.command[:,s.arm_joint_ids]).abs().mean(-1)
  quiet_phase=((s.stage>=3)&(s.forearm_force.max(-1).values<30)).float()
  return quiet_phase*(arm_speed+1.5*residual)
 if kind=='gripper_floor':
  # Tax gripper-tip load only (not wrist height — prone starts with hands near floor).
  return (s.gripper_force/25).clamp(0,1.5).mean(-1)*(s.stage<=3).float()
 if kind=='waist_assist':
  rising=((s.stage>=2)&(s.stage<=4)).float()
  wq=s.robot.data.joint_pos[:,s.waist_joint_ids]; wd=s.robot.data.joint_vel[:,s.waist_joint_ids]
  pitch_help=wd[:,2].abs().clamp(0,2)*.4+(-wq[:,2]).clamp(0,1)*.4
  return rising*s.torso_upright.clamp(0,1)*pitch_help*torch.exp(-wq[:,0].abs()/.5)
 if kind=='hip_drive':
  # Reward hip_pitch effort while rising — keep AKE90 hips engaged (not locked to prior).
  rising=((s.stage>=2)&(s.stage<=4)).float()
  hq=s.robot.data.joint_pos[:,s.hip_pitch_ids]; hd=s.robot.data.joint_vel[:,s.hip_pitch_ids].abs().mean(-1)
  return rising*hd.clamp(0,3)*.3*s.upright.clamp(0,1)
 if kind=='yaw_spin':
  rising=((s.stage>=3)&(s.stage<=4)).float()
  yaw=s.robot.data.root_ang_vel_b[:,2].abs()
  return rising*yaw
 if kind=='motor':return torch.exp(-((s.robot.data.joint_pos[:,s.pos_ids]-s.command[:,s.pos_ids])/.6).square().mean(-1))
 if kind=='unsupported_supine':return sup*early*(~s.support.all(-1)).float()*(s.height-.18).clamp(min=0)*((1.8-s.shoulder.min(-1).values).clamp(min=0)+(~s.arms.any(-1)).float())
 if kind=='airborne_prone':return prone*s.face_down.float()*s.wheel_clearance.mean(-1)
 if kind=='foot_apart':
  # Soft stand-in for missing L↔R self-collision: reward wheel center separation.
  # Nominal standing gap ≈ 0.30 m; stacking during kneel→stand drops well below that.
  centers=s.robot.data.body_pos_w[:,s.wheel_ids,:2]
  sep=(centers[:,0]-centers[:,1]).norm(dim=-1)
  return ((sep-.18)/.12).clamp(0,1)
 if kind=='hip_square':
  # Penalize adduct / 內八: L hip_roll < 0 and R hip_roll > 0 (URDF +X conventions).
  rolls=s.robot.data.joint_pos[:,s.hip_roll_ids]
  return (-rolls[:,0]).clamp(min=0)+rolls[:,1].clamp(min=0)
 raise ValueError(kind)
def timeout(env):return env.episode_length_buf*env.step_dt>=env.cfg.episode_length_s
def escaped(env):
 p=env.scene['robot'].data.root_pos_w-env.scene.env_origins
 return (p[:,:2].norm(dim=-1)>3.)|(p[:,2]<-.3)|(p[:,2]>2.)
def diagnostics(env,env_ids):
 s=state(env);s.update()
 for mode in range(2):
  ids=env_ids[(s.mode[env_ids]==mode)&(env.episode_length_buf[env_ids]>0)]
  s.totals[mode,0]+=len(ids);s.totals[mode,1]+=s.planted[ids].sum();s.totals[mode,2]+=s.knelt[ids].sum();s.totals[mode,3]+=s.upright_knelt[ids].sum();s.totals[mode,4]+=s.stood[ids].sum()
 # Unlock stand once every *active* mode has ≥3 kneels. Single-mode runs (only
 # supine or only prone) must not wait forever for the unused mode's tally.
 seen=s.totals[:,0]>0
 if seen.any() and (s.totals[seen,2]>=3).all():s.cap=4
 result={'stage_cap':s.cap}
 for m,name in enumerate(['supine','prone']):
  result[name+'_episodes']=s.totals[m,0].item()
  for i,label in enumerate(['plant_success','kneel_success','upright_kneel_success','kneel_then_stand_success'],1):result[name+'_'+label]=(s.totals[m,i]/s.totals[m,0].clamp(min=1)).item()
 return result
