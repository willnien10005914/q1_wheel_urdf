"""Open-box sit-up: supine after the crate lid opens → yoga press → stable kneel.

The stand-up PPO (``Isaac-Q1-Posture-v0``) takes over once the kneel is held. This policy does
not stand. The neck only yaws, so trunk lift comes from the waist pitch motors.
"""

from __future__ import annotations

from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as loco_mdp

import wheel_humanoid_lab.tasks.manager_based.skate.mdp as mdp
from wheel_humanoid_lab.assets import KNEEL_HIP_PITCH, KNEEL_KNEE, KNEEL_PELVIS_Z, KNEEL_WAIST_PITCH
from wheel_humanoid_lab.tasks.manager_based.posture.posture_env_cfg import POSTURE_POS_ACTION_SCALE
from wheel_humanoid_lab.tasks.manager_based.skate.skate_env_cfg import (
    EventCfg,
    ObservationsCfg,
    Q1SkateEnvCfg,
    RewardsCfg,
    _position_action_clip,
)

# Arms reach behind the torso (high shoulder pitch = +Y swing back) and stay there through the
# sit-up, then come forward for balance once the knee rollers are planted.
LIE_POSE = {
    ".*_hip_pitch_joint": -0.4,
    ".*_hip_roll_joint": 0.0,
    ".*_knee_joint": 0.4,
    ".*_knee_roller_joint": 0.1777776,
    "waist_pitch_joint": 0.0,
    ".*_shoulder_pitch_joint": 2.2,
    "l_shoulder_roll_joint": 0.3,
    "r_shoulder_roll_joint": -0.3,
    ".*_elbow_joint": -1.6,
}
TUCK_POSE = {
    ".*_hip_pitch_joint": -1.15,
    ".*_hip_roll_joint": 0.0,
    ".*_knee_joint": 2.2,
    ".*_knee_roller_joint": 0.9777768000000001,
    "waist_pitch_joint": 0.0,
    ".*_shoulder_pitch_joint": 2.2,
    "l_shoulder_roll_joint": 0.3,
    "r_shoulder_roll_joint": -0.3,
    ".*_elbow_joint": -1.6,
}
YOGA_POSE = {
    ".*_hip_pitch_joint": -0.3254,
    ".*_hip_roll_joint": 0.0,
    ".*_knee_joint": 1.3427,
    ".*_knee_roller_joint": 0.5968,
    "waist_pitch_joint": 0.5,
    ".*_shoulder_pitch_joint": 2.2,
    "l_shoulder_roll_joint": 0.2,
    "r_shoulder_roll_joint": -0.2,
    ".*_elbow_joint": -1.4,
}
KNEEL_POSE = {
    ".*_hip_pitch_joint": -1.02,
    ".*_hip_roll_joint": 0.0,
    ".*_knee_joint": 2.36,
    ".*_knee_roller_joint": 1.04888784,
    "waist_pitch_joint": 0.3,
    ".*_shoulder_pitch_joint": 0.55,
    "l_shoulder_roll_joint": 0.2,
    "r_shoulder_roll_joint": -0.2,
    ".*_elbow_joint": -0.5,
}

# Timed keyframes only go as far as the kneel. Stand is a different PPO.
UNBOX_KNOTS = [
    (0.00, LIE_POSE, 0.22),
    (0.25, TUCK_POSE, 0.28),
    (0.50, YOGA_POSE, 0.34),
    (0.75, YOGA_POSE, 0.39),
    (1.00, KNEEL_POSE, KNEEL_PELVIS_Z),
]

UNBOX_POS_ACTION_SCALE = {
    **POSTURE_POS_ACTION_SCALE,
    ".*_shoulder_pitch_joint": 2.2,
    ".*_shoulder_roll_joint": 0.8,
    ".*_elbow_joint": 1.6,
    "waist_pitch_joint": 0.8,
}

# Later-stage weights stay small until tuck + yoga actually lift the trunk.
UNBOX_STAGES = [
    {
        "iter": 0,
        "rewards": {
            "unbox_tuck": 1.8,
            "unbox_yoga_press": 2.2,
            "unbox_roller_plant": 0.5,
            "unbox_wheel_scoot": 0.0,
            "unbox_waist_rise": 0.3,
            "unbox_kneel_hold": 0.0,
        },
    },
    {
        "iter": 600,
        "rewards": {
            "unbox_roller_plant": 2.0,
            "unbox_wheel_scoot": 0.8,
            "unbox_waist_rise": 1.2,
        },
    },
    {
        "iter": 1800,
        "rewards": {
            "unbox_wheel_scoot": 1.6,
            "unbox_waist_rise": 2.0,
            "unbox_kneel_hold": 1.2,
        },
    },
    {
        "iter": 3200,
        "rewards": {"unbox_kneel_hold": 2.6},
    },
]

_HIPS = SceneEntityCfg("robot", joint_names=[".*_hip_pitch_joint"], preserve_order=True)
_KNEES = SceneEntityCfg("robot", joint_names=[".*_knee_joint"], preserve_order=True)
_SHOULDERS = SceneEntityCfg("robot", joint_names=[".*_shoulder_pitch_joint"], preserve_order=True)
_WAIST = SceneEntityCfg("robot", joint_names=["waist_pitch_joint"], preserve_order=True)
_WHEELS_J = SceneEntityCfg("robot", joint_names=["l_wheel_joint", "r_wheel_joint"], preserve_order=True)
_WHEELS_B = SceneEntityCfg("contact_forces", body_names=["l_wheel_link", "r_wheel_link"], preserve_order=True)
_ROLLERS_B = SceneEntityCfg("contact_forces", body_names=["l_knee_roller_link", "r_knee_roller_link"], preserve_order=True)


@configclass
class UnboxRewardsCfg(RewardsCfg):
    getup_track = RewTerm(func=mdp.getup_track, weight=1.6, params={"knots": UNBOX_KNOTS, "phase_s": 6.0})
    unbox_tuck = RewTerm(
        func=mdp.unbox_tuck,
        weight=1.8,
        params={"hip_cfg": _HIPS, "knee_cfg": _KNEES, "hip_target": -1.15, "knee_target": 2.20},
    )
    unbox_yoga_press = RewTerm(
        func=mdp.unbox_yoga_press,
        weight=2.2,
        params={"shoulder_cfg": _SHOULDERS, "shoulder_target": 2.2},
    )
    unbox_roller_plant = RewTerm(
        func=mdp.unbox_roller_plant,
        weight=0.5,
        params={"roller_cfg": _ROLLERS_B},
    )
    unbox_wheel_scoot = RewTerm(
        func=mdp.unbox_wheel_scoot,
        weight=0.0,
        params={"wheel_cfg": _WHEELS_J, "roller_cfg": _ROLLERS_B},
    )
    unbox_waist_rise = RewTerm(
        func=mdp.unbox_waist_rise,
        weight=0.3,
        params={
            "roller_cfg": _ROLLERS_B,
            "shoulder_cfg": _SHOULDERS,
            "waist_cfg": _WAIST,
            "shoulder_target": 0.55,
            "waist_target": KNEEL_WAIST_PITCH,
        },
    )
    unbox_kneel_hold = RewTerm(
        func=mdp.unbox_kneel_hold,
        weight=0.0,
        params={
            "wheel_cfg": _WHEELS_B,
            "roller_cfg": _ROLLERS_B,
            "hip_cfg": _HIPS,
            "knee_cfg": _KNEES,
            "kneel_z": KNEEL_PELVIS_Z,
            "hip_target": KNEEL_HIP_PITCH,
            "knee_target": KNEEL_KNEE,
        },
    )


@configclass
class UnboxObservationsCfg(ObservationsCfg):
    def __post_init__(self):
        self.policy.command.func = mdp.getup_command
        self.policy.command.params = {"command_name": "base_velocity", "arm_style": 1.0, "phase_s": 6.0}
        self.critic.command.func = mdp.getup_command
        self.critic.command.params = {"command_name": "base_velocity", "arm_style": 1.0, "phase_s": 6.0}


@configclass
class UnboxEventCfg(EventCfg):
    reset_supine = EventTerm(
        func=mdp.reset_supine,
        mode="reset",
        params={"lie_pose": LIE_POSE, "root_z": 0.22, "joint_noise": 0.04},
    )


@configclass
class UnboxTerminationsCfg:
    time_out = DoneTerm(func=loco_mdp.time_out, time_out=True)


@configclass
class UnboxCurriculumCfg:
    stage = CurrTerm(func=mdp.skate_stage, params={"stages": UNBOX_STAGES, "steps_per_iter": 24, "start_iter": 0, "force_stage": None})
    metrics = CurrTerm(
        func=mdp.unbox_metrics,
        params={
            "hip_cfg": _HIPS,
            "knee_cfg": _KNEES,
            "shoulder_cfg": _SHOULDERS,
            "wheel_joint_cfg": _WHEELS_J,
            "wheel_body_cfg": _WHEELS_B,
            "roller_cfg": _ROLLERS_B,
        },
    )


@configclass
class Q1UnboxEnvCfg(Q1SkateEnvCfg):
    observations: UnboxObservationsCfg = UnboxObservationsCfg()
    events: UnboxEventCfg = UnboxEventCfg()
    rewards: UnboxRewardsCfg = UnboxRewardsCfg()
    terminations: UnboxTerminationsCfg = UnboxTerminationsCfg()
    curriculum: UnboxCurriculumCfg = UnboxCurriculumCfg()

    def __post_init__(self):
        super().__post_init__()
        self.episode_length_s = 10.0
        self.actions.joint_pos.scale = UNBOX_POS_ACTION_SCALE
        self.actions.joint_pos.clip = _position_action_clip(UNBOX_POS_ACTION_SCALE)
        self.commands.base_velocity.rel_standing_envs = 1.0
        self.commands.base_velocity.resampling_time_range = (1.0e9, 1.0e9)
        self.events.push_robot = None

        r = self.rewards
        for name in (
            "wheel_speed",
            "base_vx_track",
            "heading_hold",
            "leg_symmetry",
            "grounded",
            "skating_air_time",
            "forward_lean",
            "arms_back",
            "base_height",
            "long_air",
            "flying",
            "wheel_slip",
            "wheel_lateral_slip",
            "undesired_contacts",
            "roller_contact",
            "lin_vel_z_l2",
        ):
            getattr(r, name).weight = 0.0
        r.action_rate_l2.weight = -0.03
        r.upright.weight = 0.0
        r.head_up.weight = 0.0
        r.alive.weight = 0.2
        r.action_l2.weight = -0.004


@configclass
class Q1UnboxEnvCfg_PLAY(Q1UnboxEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.episode_length_s = 120.0
        self.observations.policy.enable_corruption = False
        self.events.body_mass = None
        self.events.randomize_com = None
        self.events.reset_base.params["pose_range"] = {"x": (0.0, 0.0), "y": (0.0, 0.0), "yaw": (0.0, 0.0)}
        self.events.reset_robot_joints.params["position_range"] = (0.0, 0.0)
        self.events.reset_robot_joints.params["velocity_range"] = (0.0, 0.0)
        self.curriculum.stage.params["force_stage"] = len(UNBOX_STAGES) - 1
        self.viewer.eye = (1.6, -2.1, 0.20)
        self.viewer.lookat = (0.0, 0.0, -0.05)
