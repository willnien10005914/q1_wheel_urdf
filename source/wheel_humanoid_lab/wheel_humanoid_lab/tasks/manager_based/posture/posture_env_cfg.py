"""Q1 posture task: learn the kneel (four wheels) <-> stand (two wheels) transition.

A separate PPO from the skate policy, sharing the 92-D obs / 24-D action contract. The posture target
lives in ``body[0]`` of the command block (0 = stand, 1 = kneel) and flips every few seconds, so one
network learns both directions plus holding either pose. The skate policy is never retrained.
"""

from __future__ import annotations

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as loco_mdp

import wheel_humanoid_lab.tasks.manager_based.skate.mdp as mdp
from wheel_humanoid_lab.assets import (
    KNEEL_HIP_PITCH,
    KNEEL_KNEE,
    KNEEL_PELVIS_Z,
    KNEEL_WAIST_PITCH,
    Q1_POSITION_JOINTS,
    SKATE_PELVIS_Z,
)
from wheel_humanoid_lab.tasks.manager_based.skate.skate_env_cfg import (
    _POS_ACTION_SCALE,
    CommandsCfg,
    EventCfg,
    ObservationsCfg,
    Q1SkateEnvCfg,
    RewardsCfg,
    _position_action_clip,
)

# Kneel keyframe on the leg / waist joints (everything else stays at the skate default).
KNEEL_POSE = {
    ".*_hip_pitch_joint": KNEEL_HIP_PITCH,
    ".*_hip_roll_joint": 0.0,
    ".*_knee_joint": KNEEL_KNEE,
    ".*_knee_roller_joint": 0.444444 * KNEEL_KNEE,
    "waist_pitch_joint": KNEEL_WAIST_PITCH,
}
POSTURE_JOINTS = [".*_hip_pitch_joint", ".*_hip_roll_joint", ".*_knee_joint", "waist_pitch_joint"]

# The skate clip (default +- 0.45 rad on the knee) cannot reach knee ~2.35 / hip ~-1.05; widen the
# leg range (clip = default -/+ scale: knee [-1.15, 2.55], hip pitch [-1.20, 0.50]).
POSTURE_POS_ACTION_SCALE = {
    **_POS_ACTION_SCALE,
    ".*_hip_pitch_joint": 0.85,
    ".*_knee_joint": 1.85,
    "waist_pitch_joint": 0.40,
}

_ILLEGAL_BODIES_KNEEL = ["pelvis", "torso", "head_link", "waist_.*", ".*_shoulder_.*", ".*_elbow_.*", ".*_wrist_.*", ".*_gripper_.*"]
_WHEELS_B = SceneEntityCfg("contact_forces", body_names=["l_wheel_link", "r_wheel_link"], preserve_order=True)
_ROLLERS_B = SceneEntityCfg("contact_forces", body_names=["l_knee_roller_link", "r_knee_roller_link"], preserve_order=True)


@configclass
class PostureCommandsCfg(CommandsCfg):
    posture = mdp.PostureCommandCfg(resampling_time_range=(2.5, 5.0), p_kneel=0.5)


@configclass
class PostureObservationsCfg(ObservationsCfg):
    def __post_init__(self):
        self.policy.command.params["posture_command_name"] = "posture"
        self.critic.command.params["posture_command_name"] = "posture"


@configclass
class PostureEventCfg(EventCfg):
    # runs after reset_base / reset_robot_joints (definition order): half the episodes start kneeling
    reset_posture = EventTerm(
        func=mdp.reset_posture_pose,
        mode="reset",
        params={"kneel_pose": KNEEL_POSE, "kneel_root_z": KNEEL_PELVIS_Z + 0.01, "p_kneel": 0.5},
    )


@configclass
class PostureRewardsCfg(RewardsCfg):
    posture_height = RewTerm(
        func=mdp.posture_height, weight=2.0,
        params={"command_name": "posture", "stand_z": SKATE_PELVIS_Z, "kneel_z": KNEEL_PELVIS_Z, "std": 0.06},
    )
    posture_keyframe = RewTerm(
        func=mdp.posture_keyframe, weight=1.5,
        params={"command_name": "posture", "asset_cfg": SceneEntityCfg("robot", joint_names=POSTURE_JOINTS), "kneel_pose": KNEEL_POSE, "std": 0.35},
    )
    posture_contacts = RewTerm(
        func=mdp.posture_contacts, weight=1.0,
        params={"command_name": "posture", "wheel_cfg": _WHEELS_B, "roller_cfg": _ROLLERS_B, "threshold": 5.0},
    )
    stationary_base = RewTerm(func=mdp.stationary_base, weight=0.0, params={"std": 0.3})
    posture_still = RewTerm(
        func=mdp.posture_still, weight=0.5,
        params={"posture_command_name": "posture", "velocity_command_name": "base_velocity", "std": 0.3},
    )
    kneel_drive = RewTerm(
        func=mdp.kneel_drive, weight=1.6,
        params={"posture_command_name": "posture", "velocity_command_name": "base_velocity", "std_vx": 0.25, "std_yaw": 0.35},
    )


@configclass
class PostureTerminationsCfg:
    time_out = DoneTerm(func=loco_mdp.time_out, time_out=True)
    base_contact = DoneTerm(
        func=loco_mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=["pelvis", "torso", "head_link", ".*_shoulder_.*", ".*_elbow_.*"]), "threshold": 1.0},
    )
    bad_orientation = DoneTerm(func=loco_mdp.bad_orientation, params={"limit_angle": 0.7})


@configclass
class Q1PostureEnvCfg(Q1SkateEnvCfg):
    commands: PostureCommandsCfg = PostureCommandsCfg()
    observations: PostureObservationsCfg = PostureObservationsCfg()
    events: PostureEventCfg = PostureEventCfg()
    rewards: PostureRewardsCfg = PostureRewardsCfg()
    terminations: PostureTerminationsCfg = PostureTerminationsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.episode_length_s = 10.0

        # wider leg range so the kneel is reachable
        self.actions.joint_pos.scale = POSTURE_POS_ACTION_SCALE
        self.actions.joint_pos.clip = _position_action_clip(POSTURE_POS_ACTION_SCALE)

        # While kneeling, track a slow WASD command. Standing episodes still see the command in the
        # observation, but posture_still / kneel_drive make only the kneel bit follow it.
        self.commands.base_velocity.rel_standing_envs = 0.2
        self.commands.base_velocity.resampling_time_range = (3.0, 6.0)
        self.commands.base_velocity.ranges.lin_vel_x = (-0.35, 0.80)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.50, 0.50)

        # no skate curriculum; fixed weights below
        self.curriculum.stage = None
        self.curriculum.action_rate = None
        self.events.push_robot = None

        r = self.rewards
        r.wheel_speed.weight = 0.3
        r.base_vx_track.weight = 0.0
        r.heading_hold.weight = 0.0
        r.leg_symmetry.weight = 0.5
        r.grounded.weight = 0.0
        r.skating_air_time.weight = 0.0
        r.forward_lean.weight = 0.0
        r.arms_back.weight = 0.3
        r.base_height.weight = 0.0  # replaced by posture_height
        r.action_rate_l2.weight = -0.3
        r.lin_vel_z_l2.weight = -0.1
        r.long_air.weight = 0.0
        r.flying.weight = -1.0
        r.wheel_slip.weight = -0.1
        r.wheel_lateral_slip.weight = -0.2
        r.undesired_contacts.params["sensor_cfg"] = SceneEntityCfg("contact_forces", body_names=_ILLEGAL_BODIES_KNEEL)
        # rollers are supposed to touch while kneeling
        r.roller_contact.func = mdp.roller_contact_when_standing
        r.roller_contact.params = {"command_name": "posture", "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_knee_roller_link"), "threshold": 1.0}
        r.roller_contact.weight = -1.0


@configclass
class Q1PostureEnvCfg_PLAY(Q1PostureEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.scene.env_spacing = 2.5
        self.episode_length_s = 120.0
        self.observations.policy.enable_corruption = False
        self.events.body_mass = None
        self.events.randomize_com = None
        self.events.reset_base.params["pose_range"] = {"x": (0.0, 0.0), "y": (0.0, 0.0), "yaw": (0.0, 0.0)}
        self.events.reset_robot_joints.params["position_range"] = (0.0, 0.0)
        self.events.reset_robot_joints.params["velocity_range"] = (0.0, 0.0)
        self.events.reset_posture.params["p_kneel"] = 0.0
        self.commands.posture.resampling_time_range = (1.0e9, 1.0e9)
        self.viewer.eye = (1.9, -2.3, 0.35)
        self.viewer.lookat = (0.0, 0.0, -0.15)
