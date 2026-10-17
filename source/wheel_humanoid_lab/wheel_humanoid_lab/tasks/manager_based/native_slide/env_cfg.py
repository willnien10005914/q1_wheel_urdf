from isaaclab.utils import configclass
from isaaclab.managers import RewardTermCfg as RewTerm, CurriculumTermCfg as CurrTerm, TerminationTermCfg as DoneTerm
from ..inertia.native_env_cfg import NativeStandEnvCfg, NativeWebEnvCfg, CommandsCfg, ActionsCfg, RewardsCfg, CurriculumCfg, TerminationsCfg
from . import mdp


@configclass
class SlideCommandsCfg(CommandsCfg):
    skill = mdp.SlideCommandCfg()


@configclass
class SlideActionsCfg(ActionsCfg):
    motors = mdp.SlideActionCfg()


@configclass
class SlideRewardsCfg(RewardsCfg):
    stand = RewTerm(func=mdp.base.stand_reward, weight=.2)
    loaded_backward_push = RewTerm(func=mdp.push, weight=4.)
    airborne_forward_return = RewTerm(func=mdp.recover, weight=1.)
    completed_push_return = RewTerm(func=mdp.cycle, weight=6.)
    hopping = RewTerm(func=mdp.hop, weight=-2.)


@configclass
class SlideCurriculumCfg(CurriculumCfg):
    skill = CurrTerm(func=mdp.curriculum, params={'start_iter':0, 'steps_per_iteration':24})


@configclass
class SlideTerminationsCfg(TerminationsCfg):
    failure = DoneTerm(func=mdp.failure)


@configclass
class NativeSlideEnvCfg(NativeStandEnvCfg):
    commands = SlideCommandsCfg()
    actions = SlideActionsCfg()
    rewards = SlideRewardsCfg()
    curriculum = SlideCurriculumCfg()
    terminations = SlideTerminationsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.commands.skill.mode = 'walk'
        self.commands.base_velocity.ranges.lin_vel_x = (.25, .6)
        self.commands.base_velocity.ranges.ang_vel_z = (-.15, .15)
        self.commands.base_velocity.rel_standing_envs = .15


@configclass
class NativeSlideWebEnvCfg(NativeWebEnvCfg):
    commands = SlideCommandsCfg()
    actions = SlideActionsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.commands.skill.sliding = False

