"""Independent video-motion PPO tasks; old models retain their own environments."""
from isaaclab.utils import configclass
from isaaclab.managers import RewardTermCfg as RewTerm,EventTermCfg as EventTerm,CurriculumTermCfg as CurrTerm,TerminationTermCfg as DoneTerm
from ..inertia.native_env_cfg import NativeStandEnvCfg,CommandsCfg,ActionsCfg,RewardsCfg,EventsCfg,CurriculumCfg,TerminationsCfg
from . import mdp

@configclass
class ReferenceCommandsCfg(CommandsCfg):
    skill=mdp.ReferenceCommandCfg()
@configclass
class ReferenceActionsCfg(ActionsCfg):
    motors=mdp.ReferenceActionCfg()
@configclass
class ReferenceEventsCfg(EventsCfg):
    reset=EventTerm(func=mdp.reset_reference,mode='reset')
@configclass
class ReferenceRewardsCfg(RewardsCfg):
    pose=RewTerm(func=mdp.pose_tracking,weight=3.)
    root_pitch=RewTerm(func=mdp.root_pitch_tracking,weight=1.)
    height=RewTerm(func=mdp.height_tracking,weight=1.)
    stand=RewTerm(func=mdp.terminal_hold,weight=2.)
    upright=RewTerm(func=mdp.base.upright_reward,weight=1.)
    support=RewTerm(func=mdp.contacts,weight=.5)
    no_hop=RewTerm(func=mdp.no_hop,weight=-2.)
    slow_crouch=RewTerm(func=mdp.slow_crouch_cost,weight=-3.)
    landing_arms=RewTerm(func=mdp.landing_arms,weight=1.)
    push=RewTerm(func=mdp.push,weight=0.)
    recover=RewTerm(func=mdp.recover,weight=0.)
    cycles=RewTerm(func=mdp.cycles,weight=0.)
@configclass
class ReferenceCurriculumCfg(CurriculumCfg):
    skill=CurrTerm(func=mdp.curriculum,params={'start_iter':0,'steps_per_iteration':24})
@configclass
class ReferenceTerminationsCfg(TerminationsCfg):
    failure=DoneTerm(func=mdp.failure)

@configclass
class ReferenceStandupEnvCfg(NativeStandEnvCfg):
    reference_skill:str='standup'
    boot_squat:bool=False
    reference_phase_resets:bool=True
    commands=ReferenceCommandsCfg();actions=ReferenceActionsCfg();events=ReferenceEventsCfg()
    rewards=ReferenceRewardsCfg();curriculum=ReferenceCurriculumCfg();terminations=ReferenceTerminationsCfg()
    def __post_init__(self):
        super().__post_init__();self.commands.skill.mode=self.reference_skill;self.commands.skill.boot_squat=self.boot_squat
        self.episode_length_s=14.
@configclass
class ReferenceKneelEnvCfg(ReferenceStandupEnvCfg):
    reference_skill:str='kneel'
    def __post_init__(self):
        super().__post_init__()
        # Retimed source currently takes ~20s; retain a generous terminal hold
        # budget instead of truncating a slow descent at the old 8s timestamp.
        self.episode_length_s=36.
@configclass
class ReferenceWalkEnvCfg(ReferenceStandupEnvCfg):
    reference_skill:str='walk'
    def __post_init__(self):
        super().__post_init__();self.commands.base_velocity.rel_standing_envs=.2
        self.commands.base_velocity.ranges.lin_vel_x=(-.2,.4);self.commands.base_velocity.ranges.ang_vel_z=(-.15,.15)
@configclass
class ReferenceBackflipEnvCfg(ReferenceStandupEnvCfg):
    reference_skill:str='backflip'
    def __post_init__(self):
        super().__post_init__();self.episode_length_s=8.
        self.rewards.velocity.weight=0.;self.rewards.yaw.weight=0.;self.rewards.flip.weight=4.
@configclass
class ReferenceSkateEnvCfg(ReferenceWalkEnvCfg):
    def __post_init__(self):
        super().__post_init__();self.commands.skill.sliding=True
        self.commands.base_velocity.ranges.lin_vel_x=(.2,.4)
        self.rewards.push.weight=4.;self.rewards.recover.weight=1.;self.rewards.cycles.weight=6.
@configclass
class ReferenceWebEnvCfg(ReferenceStandupEnvCfg):
    reference_skill:str='kneel'
    boot_squat:bool=True
    reference_phase_resets:bool=False
    def __post_init__(self):
        super().__post_init__();self.scene.num_envs=1;self.curriculum=None
        self.observations.policy.enable_corruption=False
        self.terminations.failure=None;self.terminations.time_out=None;self.episode_length_s=3600.
