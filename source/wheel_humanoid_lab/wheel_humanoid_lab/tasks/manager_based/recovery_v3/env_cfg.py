from isaaclab.utils import configclass
from isaaclab.managers import RewardTermCfg as Rew,EventTermCfg as Event,CurriculumTermCfg as Curr,TerminationTermCfg as Done
import isaaclab_tasks.manager_based.locomotion.velocity.mdp as loco
from wheel_humanoid_lab.tasks.manager_based.recovery.recovery_env_cfg import Q1RecoveryEnvCfg,RecoveryObservationsCfg,RecoveryActionsCfg
from wheel_humanoid_lab.tasks.manager_based.skate.skate_env_cfg import EventCfg
from wheel_humanoid_lab.assets import Q1_POSITION_JOINTS,Q1_WHEEL_JOINTS
from . import mdp
@configclass
class Actions(RecoveryActionsCfg):
 wheel_vel=mdp.ContactWheelVelocityActionCfg(asset_name='robot',joint_names=Q1_WHEEL_JOINTS,preserve_order=True,scale=8.,clip={'.*_wheel_joint':(-8.,8.)})
 joint_pos=mdp.ContactPositionActionCfg(asset_name='robot',joint_names=Q1_POSITION_JOINTS,preserve_order=True,scale=1.,clip=None)
@configclass
class Observations(RecoveryObservationsCfg):
 def __post_init__(self):
  self.policy.command.func=mdp.observation;self.policy.command.params={}
  self.critic.command.func=mdp.observation;self.critic.command.params={}
@configclass
class Events(EventCfg):
 reset_reference=Event(func=mdp.reset,mode='reset',params={'mode':-1})
@configclass
class Rewards:
 approach=Rew(func=mdp.reward,weight=1.,params={'kind':'approach'})
 arm_support=Rew(func=mdp.reward,weight=4.,params={'kind':'plant'})
 wheel_support=Rew(func=mdp.reward,weight=4.,params={'kind':'wheel'})
 kneel=Rew(func=mdp.reward,weight=8.,params={'kind':'kneel'})
 stage_progress=Rew(func=mdp.reward,weight=10.,params={'kind':'progress'})
 supported_lift=Rew(func=mdp.reward,weight=2.,params={'kind':'lift'})
 stand=Rew(func=mdp.reward,weight=10.,params={'kind':'stand'})
 arm_assist=Rew(func=mdp.reward,weight=3.,params={'kind':'arm_assist'})
 arm_calm=Rew(func=mdp.reward,weight=-2.,params={'kind':'arm_calm'})
 motor_prior=Rew(func=mdp.reward,weight=.2,params={'kind':'motor'})
 unsupported_supine=Rew(func=mdp.reward,weight=-5.,params={'kind':'unsupported_supine'})
 airborne_prone=Rew(func=mdp.reward,weight=-4.,params={'kind':'airborne_prone'})
 # Soft L↔R foot spacing (self-collision is globally off) + anti-pigeon-toe.
 foot_apart=Rew(func=mdp.reward,weight=2.,params={'kind':'foot_apart'})
 hip_square=Rew(func=mdp.reward,weight=-3.,params={'kind':'hip_square'})
 action_rate=Rew(func=loco.action_rate_l2,weight=-.02)
 action_size=Rew(func=loco.action_l2,weight=-.01)
 torque=Rew(func=loco.joint_torques_l2,weight=-1.e-5)
 limits=Rew(func=loco.joint_pos_limits,weight=-1.)
 escaped=Rew(func=loco.is_terminated,weight=-5.)
@configclass
class Terminations:
 time_out=Done(func=mdp.timeout,time_out=True)
 escaped=Done(func=mdp.escaped)
@configclass
class Curriculum:
 recovery_v3=Curr(func=mdp.diagnostics)
@configclass
class Q1RecoveryV3EnvCfg(Q1RecoveryEnvCfg):
 actions:Actions=Actions()
 observations:Observations=Observations()
 events:Events=Events()
 rewards:Rewards=Rewards()
 terminations:Terminations=Terminations()
 curriculum:Curriculum=Curriculum()
 def __post_init__(self):
  super().__post_init__();self.episode_length_s=40.;self.scene.env_spacing=5.
  # Start with nominal mass/COM: motor electrical limits/delay/friction remain intact.
  self.events.body_mass=None;self.events.randomize_com=None
@configclass
class Q1RecoveryV3PlayCfg(Q1RecoveryV3EnvCfg):
 def __post_init__(self):
  super().__post_init__();self.scene.num_envs=1;self.observations.policy.enable_corruption=False

@configclass
class Q1RecoveryV3SupineEnvCfg(Q1RecoveryV3EnvCfg):
 """Independent PPO for 正躺 → kneel → stand only."""
 def __post_init__(self):
  super().__post_init__()
  self.events.reset_reference.params={'mode':0}
  # Smooth-arms mixed run stuck at kneel for supine; push stand harder relative to kneel.
  self.rewards.stand.weight=14.
  self.rewards.kneel.weight=6.
  self.rewards.stage_progress.weight=12.
  self.episode_length_s=45.

@configclass
class Q1RecoveryV3SupinePlayCfg(Q1RecoveryV3SupineEnvCfg):
 def __post_init__(self):
  super().__post_init__();self.scene.num_envs=1;self.observations.policy.enable_corruption=False

@configclass
class Q1RecoveryV3ProneEnvCfg(Q1RecoveryV3EnvCfg):
 """Independent PPO for 趴躺 → kneel → stand only."""
 def __post_init__(self):
  super().__post_init__()
  self.events.reset_reference.params={'mode':1}
  # Prone already stands well; keep stand pressure and emphasize clean foot spacing.
  self.rewards.stand.weight=12.
  self.rewards.foot_apart.weight=3.
  self.episode_length_s=40.

@configclass
class Q1RecoveryV3PronePlayCfg(Q1RecoveryV3ProneEnvCfg):
 def __post_init__(self):
  super().__post_init__();self.scene.num_envs=1;self.observations.policy.enable_corruption=False
