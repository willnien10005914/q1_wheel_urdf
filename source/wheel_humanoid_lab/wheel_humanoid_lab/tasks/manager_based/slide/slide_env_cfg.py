"""Q1 slide task: X2 push-skate. Left and right feet trade places, each one lifting briefly.

Same plant / obs / action as skate PPO so we warm-start from ``checkpoints/q1_skate_ppo.pt``.

`slide_8192envs_microlift` never lifted. `grounded` (0.69 / 0.70) and `wheel_speed` paid for
both wheels rolling, `stride_leg_split` saturated on a frozen hip difference, and
`micro_unweight` stayed at ~0.01. Mean air time was 0.4 ms.

This mix:
* no reward for both wheels planted or for a symmetric stance while a stride is commanded,
* `double_support_when_moving` costs a planted pair during forward commands,
* `rear_foot_unweight` pays only while the rearward foot is actively unloading (or briefly airborne),
* `stride_swap` pays only after the other foot becomes the rear one (a frozen split scores 0),
* `loaded_wheel_speed` matches vx on the wheel that is still down, so the lift is not a speed penalty,
* `roll_shift` pays a small left/right lean inside the same 20–220 ms window,
* `air_over_cap` still kills a parked one-wheel glide.
"""

from __future__ import annotations

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from wheel_humanoid_lab.tasks.manager_based.skate.skate_env_cfg import (
    MIN_STAND_Z,
    UPRIGHT_TILT,
    Q1SkateEnvCfg,
    RewardsCfg,
)
import wheel_humanoid_lab.tasks.manager_based.skate.mdp as mdp

_WHEELS_B = SceneEntityCfg("contact_forces", body_names=["l_wheel_link", "r_wheel_link"], preserve_order=True)
_WHEELS_J = SceneEntityCfg("robot", joint_names=["l_wheel_joint", "r_wheel_joint"], preserve_order=True)

# X2 micro-lift: a few centimetres, 20–220 ms. 20 ms is enough to count as a real unweight.
MICRO_AIR_MIN = 0.02
MICRO_AIR_MAX = 0.22

SLIDE_CMD_VX = (0.3, 1.5)
SLIDE_CMD_YAW = (-0.4, 0.4)


@configclass
class SlideRewardsCfg(RewardsCfg):
    micro_unweight = RewTerm(
        func=mdp.micro_unweight,
        weight=2.0,
        params={
            "sensor_cfg": _WHEELS_B,
            "command_name": "base_velocity",
            "min_air": MICRO_AIR_MIN,
            "max_air": MICRO_AIR_MAX,
            "max_tilt": UPRIGHT_TILT,
            "min_height": MIN_STAND_Z,
        },
    )
    stride_alternation = RewTerm(
        func=mdp.stride_alternation,
        weight=1.5,
        params={"sensor_cfg": _WHEELS_B, "command_name": "base_velocity", "min_air": MICRO_AIR_MIN},
    )
    stride_leg_split = RewTerm(
        func=mdp.stride_leg_split,
        weight=0.0,
        params={"command_name": "base_velocity", "std": 0.28},
    )
    rear_foot_unweight = RewTerm(
        func=mdp.rear_foot_unweight,
        weight=2.0,
        params={
            "sensor_cfg": _WHEELS_B,
            "command_name": "base_velocity",
            "min_split": 0.28,
            "min_air": MICRO_AIR_MIN,
            "max_air": MICRO_AIR_MAX,
            "max_tilt": UPRIGHT_TILT,
            "min_height": MIN_STAND_Z,
        },
    )
    stride_swap = RewTerm(
        func=mdp.stride_swap,
        weight=2.0,
        params={"command_name": "base_velocity", "min_split": 0.30, "swap_window": 0.9},
    )
    roll_shift = RewTerm(
        func=mdp.roll_shift,
        weight=0.5,
        params={
            "sensor_cfg": _WHEELS_B,
            "command_name": "base_velocity",
            "min_air": MICRO_AIR_MIN,
            "max_air": MICRO_AIR_MAX,
            "max_tilt": UPRIGHT_TILT,
            "min_height": MIN_STAND_Z,
        },
    )
    loaded_wheel_speed = RewTerm(
        func=mdp.loaded_wheel_speed,
        weight=0.8,
        params={
            "command_name": "base_velocity",
            "std": 0.35,
            "asset_cfg": _WHEELS_J,
            "sensor_cfg": _WHEELS_B,
        },
    )
    double_support_when_moving = RewTerm(
        func=mdp.double_support_when_moving,
        weight=-1.0,
        params={"sensor_cfg": _WHEELS_B, "command_name": "base_velocity", "threshold": 5.0, "min_cmd": 0.25},
    )
    air_over_cap = RewTerm(
        func=mdp.air_over_cap,
        weight=-5.0,
        params={"sensor_cfg": _WHEELS_B, "cap": MICRO_AIR_MAX},
    )


@configclass
class Q1SlideEnvCfg(Q1SkateEnvCfg):
    rewards: SlideRewardsCfg = SlideRewardsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.episode_length_s = 20.0
        self.sim.physx.gpu_max_rigid_patch_count = 20 * 2**15

        self.curriculum.stage = None
        self.curriculum.action_rate = None
        cmd = self.commands.base_velocity
        cmd.ranges.lin_vel_x = SLIDE_CMD_VX
        cmd.ranges.ang_vel_z = SLIDE_CMD_YAW
        cmd.rel_standing_envs = 0.05
        cmd.resampling_time_range = (3.0, 6.0)
        self.events.reset_base.params["velocity_range"]["x"] = (0.0, 0.4)
        self.events.push_robot.params["velocity_range"] = {"x": (-0.4, 0.4), "y": (-0.4, 0.4)}
        self.events.randomize_com.params["com_range"] = {"x": (-0.03, 0.03), "y": (-0.03, 0.03), "z": (-0.03, 0.03)}

        r = self.rewards
        # Skate's final action-rate weight (-0.6) freezes the legs. A stride has to move.
        r.action_rate_l2.weight = -0.15
        # Mean of both wheels made a one-foot lift look like a speed error. Score the loaded wheel.
        r.wheel_speed.weight = 0.0
        r.base_vx_track.weight = 2.0
        r.heading_hold.weight = 0.8
        r.leg_symmetry.weight = 0.0
        r.grounded.weight = 0.0
        r.skating_air_time.weight = 1.5
        r.skating_air_time.params["min_air"] = MICRO_AIR_MIN
        r.skating_air_time.params["max_air"] = MICRO_AIR_MAX
        r.forward_lean.weight = 1.0
        r.arms_back.weight = 0.8
        r.base_height.weight = -0.3
        r.long_air.weight = -4.0
        r.long_air.params["cap"] = MICRO_AIR_MAX
        r.flying.weight = -3.0
        r.wheel_slip.weight = -0.15


@configclass
class Q1SlideEnvCfg_PLAY(Q1SlideEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.scene.env_spacing = 2.5
        self.episode_length_s = 120.0
        self.observations.policy.enable_corruption = False
        self.events.push_robot = None
        self.events.body_mass = None
        self.events.randomize_com = None
        self.events.reset_base.params["pose_range"] = {"x": (0.0, 0.0), "y": (0.0, 0.0), "yaw": (0.0, 0.0)}
        self.events.reset_base.params["velocity_range"]["x"] = (0.0, 0.0)
        self.events.reset_robot_joints.params["position_range"] = (0.0, 0.0)
        self.events.reset_robot_joints.params["velocity_range"] = (0.0, 0.0)
        self.commands.base_velocity.resampling_time_range = (1.0e9, 1.0e9)
        self.commands.base_velocity.rel_standing_envs = 0.0
        self.viewer.eye = (1.9, -2.3, 0.35)
        self.viewer.lookat = (0.0, 0.0, -0.15)
