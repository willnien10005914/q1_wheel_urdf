"""One 23-action/90-observation manager contract for native Q1 skills and Web."""
import isaaclab.envs.mdp as base_mdp
from isaaclab.managers import CurriculumTermCfg as CurrTerm, EventTermCfg as EventTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm, RewardTermCfg as RewTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
from .inertia_env_cfg import Q1InertiaSkateEnvCfg, CommandsCfg as BaseCommands, ObservationsCfg as BaseObservations
from .mdp import InertiaMimicActionCfg
from .mdp import nonfinite_state
from . import native_skills as mdp
from wheel_humanoid_lab.assets.q1_inertia import (
    Q1_INERTIA_MIMIC_JOINTS, Q1_INERTIA_MIMIC_DRIVERS, Q1_INERTIA_MIMIC_MULTIPLIERS,
)


@configclass
class CommandsCfg(BaseCommands):
    skill = mdp.NativeSkillCommandCfg()


@configclass
class ActionsCfg:
    motors = mdp.NativeSkillActionCfg()
    mimic = InertiaMimicActionCfg(asset_name="robot", follower_joints=Q1_INERTIA_MIMIC_JOINTS,
                                driver_joints=Q1_INERTIA_MIMIC_DRIVERS, multipliers=Q1_INERTIA_MIMIC_MULTIPLIERS)


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(BaseObservations.PolicyCfg):
        skill = ObsTerm(func=base_mdp.generated_commands, params={"command_name": "skill"})
        base_lin_vel = ObsTerm(func=base_mdp.base_lin_vel)
        height = ObsTerm(func=base_mdp.base_pos_z)

    @configclass
    class CriticCfg(BaseObservations.CriticCfg):
        skill = ObsTerm(func=base_mdp.generated_commands, params={"command_name": "skill"})
        height = ObsTerm(func=base_mdp.base_pos_z)

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class EventsCfg:
    reset = EventTerm(func=mdp.reset_native, mode="reset")


@configclass
class RewardsCfg:
    alive = RewTerm(func=base_mdp.is_alive, weight=1.)
    upright = RewTerm(func=mdp.upright_reward, weight=2.)
    height = RewTerm(func=mdp.skill_height, weight=1.)
    pose = RewTerm(func=mdp.pose_tracking, weight=1.)
    stand = RewTerm(func=mdp.stand_reward, weight=3.)
    flip = RewTerm(func=mdp.flip_reward, weight=2.)
    velocity = RewTerm(func=base_mdp.track_lin_vel_xy_exp, weight=2., params={"command_name": "base_velocity", "std": .5})
    yaw = RewTerm(func=base_mdp.track_ang_vel_z_exp, weight=.5, params={"command_name": "base_velocity", "std": .5})
    action_rate = RewTerm(func=base_mdp.action_rate_l2, weight=-.03)
    action_size = RewTerm(func=base_mdp.action_l2, weight=-.01)
    joint_speed = RewTerm(func=base_mdp.joint_vel_l2, weight=-.0001)
    limits = RewTerm(func=base_mdp.joint_pos_limits, weight=-1.)
    termination = RewTerm(func=base_mdp.is_terminated, weight=-4.)


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=base_mdp.time_out, time_out=True)
    failure = DoneTerm(func=mdp.physical_failure)
    nonfinite = DoneTerm(func=nonfinite_state)


@configclass
class CurriculumCfg:
    skill = CurrTerm(func=mdp.native_curriculum, params={"start_iter": 0, "steps_per_iteration": 24})


@configclass
class NativeStandEnvCfg(Q1InertiaSkateEnvCfg):
    native_skill: str = "stand"
    commands: CommandsCfg = CommandsCfg()
    actions: ActionsCfg = ActionsCfg()
    observations: ObservationsCfg = ObservationsCfg()
    events: EventsCfg = EventsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    curriculum: CurriculumCfg = CurriculumCfg()
    reference_phase_resets: bool = True

    def __post_init__(self):
        super().__post_init__()
        self.commands.skill.mode = self.native_skill
        self.commands.base_velocity.ranges.lin_vel_x = (0., 0.)
        self.commands.base_velocity.ranges.ang_vel_z = (0., 0.)
        self.commands.base_velocity.rel_standing_envs = 1.
        self.scene.num_envs = 1024


@configclass
class NativeWalkEnvCfg(NativeStandEnvCfg):
    native_skill: str = "walk"
    def __post_init__(self):
        super().__post_init__()
        self.commands.base_velocity.rel_standing_envs = .2


@configclass
class NativeStandupEnvCfg(NativeStandEnvCfg):
    native_skill: str = "standup"
    def __post_init__(self):
        super().__post_init__()
        self.episode_length_s = 10.


@configclass
class NativeBackflipEnvCfg(NativeStandEnvCfg):
    native_skill: str = "backflip"
    def __post_init__(self):
        super().__post_init__()
        self.episode_length_s = 6.
        self.rewards.velocity.weight = 0.
        self.rewards.yaw.weight = 0.
        self.rewards.joint_speed.weight = -.00001


@configclass
class NativeWebEnvCfg(NativeStandEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.observations.policy.enable_corruption = False
        self.commands.base_velocity.resampling_time_range = (1.e9, 1.e9)
        self.curriculum = None
        self.terminations.failure = None
        self.terminations.time_out = None
        self.episode_length_s = 3600.
