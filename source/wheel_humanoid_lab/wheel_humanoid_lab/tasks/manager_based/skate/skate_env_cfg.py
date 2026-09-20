"""Q1 Wheel inline-skate / AgiBot-X2 swizzle task (Isaac Lab manager-based env).

Plant: 24 CubeMars actuators (explicit BLDC model), Xsens MTi-630 on the pelvis, 200 Hz PhysX,
50 Hz policy. Observation/action contract: see ``wheel_humanoid_lab.assets.q1_spec``.
"""

from __future__ import annotations

import math

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg, ViewerCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, ImuCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveGaussianNoiseCfg as Gnoise

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as loco_mdp

import wheel_humanoid_lab.tasks.manager_based.skate.mdp as mdp
from wheel_humanoid_lab.assets import (
    DECIMATION,
    IMU_CFG,
    PHYSICS_DT,
    Q1_JOINT_ORDER,
    Q1_LEG_PAIRS,
    Q1_POSITION_JOINTS,
    Q1_WHEEL_CUBEMARS_CFG,
    Q1_WHEEL_JOINTS,
    SKATE_PELVIS_Z,
    WHEEL_RADIUS,
)

NUM_STEPS_PER_ENV = 24
UPRIGHT_TILT = 0.45  # rad, gate for lean / arms / air-time rewards (~26 deg)
MIN_STAND_Z = 0.65  # m, gate for the same rewards

# ------------------------------------------------------------------------------------------------
# Curriculum table (iteration -> settings). Do not train the full style in one reward dump.
# ------------------------------------------------------------------------------------------------
SKATE_STAGES = [
    {  # 0: stand / balance on two wheels
        "iter": 0,
        "cmd_vx": (0.0, 0.0), "cmd_yaw": (0.0, 0.0), "rel_standing": 1.0, "reset_vx": (0.0, 0.0),
        "push": None, "com": 0.0,
        "rewards": {"wheel_speed": 1.0, "base_vx_track": 1.0, "heading_hold": 0.0, "leg_symmetry": 0.3,
                    "grounded": 1.0, "skating_air_time": 0.0, "forward_lean": 0.0, "arms_back": 0.0,
                    "base_height": -1.0},
    },
    {  # 1: wheel-driven glide, both wheels down, no lean bonus yet
        "iter": 300,
        "cmd_vx": (0.0, 0.6), "cmd_yaw": (0.0, 0.0), "rel_standing": 0.2, "reset_vx": (0.0, 0.2),
        "rewards": {"wheel_speed": 2.0, "base_vx_track": 2.0, "grounded": 1.0, "base_height": -1.0},
    },
    {  # 2: swizzle (leg symmetry + free hip roll) at 0.3-0.8 m/s, gentle braking
        "iter": 900,
        "cmd_vx": (-0.3, 0.8), "cmd_yaw": (-0.3, 0.3), "rel_standing": 0.15, "reset_vx": (0.0, 0.3),
        "rewards": {"leg_symmetry": 1.0, "heading_hold": 0.5, "base_height": -0.3},
    },
    {  # 3: forward lean + arms back
        "iter": 1600,
        "cmd_vx": (-0.4, 1.0), "reset_vx": (0.0, 0.4),
        "rewards": {"forward_lean": 1.0, "arms_back": 0.8},
    },
    {  # 4: allow micro-lift, faster
        "iter": 2400,
        "cmd_vx": (-0.5, 1.5), "cmd_yaw": (-0.5, 0.5),
        "rewards": {"skating_air_time": 0.3},
    },
    {  # 5: pushes, CoM DR, heading
        "iter": 3400,
        "cmd_vx": (-0.6, 1.8), "cmd_yaw": (-0.6, 0.6), "push": (-0.5, 0.5), "com": 0.03,
        "rewards": {"heading_hold": 1.0},
    },
]
ACTION_RATE_KNOTS = [(0, -0.05), (1500, -0.3), (4000, -0.6)]

_ILLEGAL_BODIES = ["pelvis", "torso", "head_link", "waist_.*", ".*_shoulder_.*", ".*_elbow_.*", ".*_wrist_.*", ".*_gripper_.*", ".*_thigh_link", ".*_hip_pitch_link"]
_UNUSED_JOINTS = ["neck_joint", ".*_wrist_joint", ".*_gripper_joint", "waist_yaw_joint", "waist_roll_joint"]
_ARM_KEYFRAME_JOINTS = [".*_shoulder_pitch_joint", ".*_shoulder_roll_joint", ".*_elbow_joint"]
_AKE90_GROUPS = ["ake90_hip", "ake90_knee"]


##
# Scene
##
@configclass
class Q1SkateSceneCfg(InteractiveSceneCfg):
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        terrain_generator=None,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=0.9,
        ),
        debug_vis=False,
    )
    robot: ArticulationCfg = Q1_WHEEL_CUBEMARS_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True)
    imu = ImuCfg(
        prim_path="{ENV_REGEX_NS}/Robot/" + IMU_CFG["parent_link"],
        offset=ImuCfg.OffsetCfg(pos=tuple(IMU_CFG["xyz"])),
        update_period=0.0,
        debug_vis=False,
    )
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )


##
# MDP
##
@configclass
class CommandsCfg:
    base_velocity = loco_mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(6.0, 10.0),
        rel_standing_envs=1.0,
        rel_heading_envs=0.0,
        heading_command=False,
        debug_vis=True,
        ranges=loco_mdp.UniformVelocityCommandCfg.Ranges(lin_vel_x=(0.0, 0.0), lin_vel_y=(0.0, 0.0), ang_vel_z=(0.0, 0.0)),
    )


@configclass
class ActionsCfg:
    """24-D action = [22 position targets in contract order] + [2 wheel velocity targets]."""

    joint_pos = loco_mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=Q1_POSITION_JOINTS,
        preserve_order=True,
        scale={
            "waist_yaw_joint": 0.15,
            "waist_roll_joint": 0.10,
            "waist_pitch_joint": 0.20,
            "neck_joint": 0.20,
            ".*_shoulder_pitch_joint": 0.40,
            ".*_shoulder_roll_joint": 0.25,
            ".*_shoulder_yaw_joint": 0.20,
            ".*_elbow_joint": 0.30,
            ".*_wrist_joint": 0.0,  # frozen for the skate task, slot kept for hot-swap
            ".*_gripper_joint": 0.0,
            ".*_hip_pitch_joint": 0.50,
            ".*_hip_roll_joint": 0.30,
            ".*_knee_joint": 0.45,
        },
        use_default_offset=True,
    )
    wheel_vel = loco_mdp.JointVelocityActionCfg(
        asset_name="robot", joint_names=Q1_WHEEL_JOINTS, preserve_order=True, scale=25.0, use_default_offset=True
    )


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        """Actor: 78 proprio + 14 command = 92. Hot-swappable layout, unused command slots zero padded."""

        imu_gyro = ObsTerm(
            func=mdp.ImuObs,
            params={"sensor_cfg": SceneEntityCfg("imu"), "quantity": "ang_vel", "max_delay": IMU_CFG["sim"]["delay_steps"][1] // 2 + 1, "tilt_deg": IMU_CFG["sim"]["tilt_dr_deg"]},
            noise=Gnoise(std=IMU_CFG["sim"]["gyro_noise_std_rad_s"]),
        )
        imu_projected_gravity = ObsTerm(
            func=mdp.ImuObs,
            params={"sensor_cfg": SceneEntityCfg("imu"), "quantity": "projected_gravity", "max_delay": IMU_CFG["sim"]["delay_steps"][1] // 2 + 1, "tilt_deg": IMU_CFG["sim"]["tilt_dr_deg"]},
            noise=Gnoise(std=IMU_CFG["sim"]["grav_noise_std"]),
        )
        joint_pos = ObsTerm(
            func=mdp.JointPosContract,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=Q1_JOINT_ORDER, preserve_order=True), "bias_deg": 1.5},
            noise=Gnoise(std=0.005),
        )
        joint_vel = ObsTerm(
            func=mdp.JointVelContract,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=Q1_JOINT_ORDER, preserve_order=True), "delay": 1},
            noise=Gnoise(std=0.3),
        )
        last_action = ObsTerm(func=loco_mdp.last_action)
        command = ObsTerm(func=mdp.skate_command, params={"command_name": "base_velocity", "arm_style": 1.0})

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class CriticCfg(ObsGroup):
        """Privileged critic: clean proprio + base lin vel, wheel forces/slip/air time."""

        base_lin_vel = ObsTerm(func=loco_mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=loco_mdp.base_ang_vel)
        projected_gravity = ObsTerm(func=loco_mdp.projected_gravity)
        base_height = ObsTerm(func=loco_mdp.base_pos_z)
        joint_pos = ObsTerm(func=mdp.joint_pos_contract_clean, params={"asset_cfg": SceneEntityCfg("robot", joint_names=Q1_JOINT_ORDER, preserve_order=True)})
        joint_vel = ObsTerm(func=loco_mdp.joint_vel, params={"asset_cfg": SceneEntityCfg("robot", joint_names=Q1_JOINT_ORDER, preserve_order=True)})
        last_action = ObsTerm(func=loco_mdp.last_action)
        command = ObsTerm(func=mdp.skate_command, params={"command_name": "base_velocity", "arm_style": 1.0})
        wheel_rim_speed = ObsTerm(func=mdp.wheel_rim_speed, params={"asset_cfg": SceneEntityCfg("robot", joint_names=Q1_WHEEL_JOINTS, preserve_order=True)})
        wheel_force = ObsTerm(func=mdp.wheel_contact_normal_force, params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=["l_wheel_link", "r_wheel_link"], preserve_order=True)})
        wheel_air_time = ObsTerm(func=mdp.wheel_air_time, params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=["l_wheel_link", "r_wheel_link"], preserve_order=True)})

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class EventCfg:
    # -- startup DR
    wheel_material = EventTerm(
        func=loco_mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_wheel_link"),
            "static_friction_range": (0.6, 1.3),
            "dynamic_friction_range": (0.5, 1.1),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )
    body_mass = EventTerm(
        func=loco_mdp.randomize_rigid_body_mass,
        mode="startup",
        params={"asset_cfg": SceneEntityCfg("robot", body_names=".*"), "mass_distribution_params": (0.95, 1.05), "operation": "scale"},
    )
    joint_armature = EventTerm(
        func=loco_mdp.randomize_joint_parameters,
        mode="startup",
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=Q1_JOINT_ORDER), "armature_distribution_params": (0.9, 1.1), "operation": "scale"},
    )
    # -- reset (restore-then-sample)
    randomize_com = EventTerm(
        func=mdp.randomize_com,
        mode="reset",
        params={"asset_cfg": SceneEntityCfg("robot", body_names=["pelvis", "torso"]), "com_range": {"x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0)}},
    )
    reset_base = EventTerm(
        func=loco_mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.3, 0.3), "y": (-0.3, 0.3), "yaw": (-math.pi, math.pi)},
            "velocity_range": {"x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0), "roll": (0.0, 0.0), "pitch": (0.0, 0.0), "yaw": (0.0, 0.0)},
        },
    )
    reset_robot_joints = EventTerm(
        func=loco_mdp.reset_joints_by_offset,
        mode="reset",
        params={"position_range": (-0.05, 0.05), "velocity_range": (-0.1, 0.1), "asset_cfg": SceneEntityCfg("robot", joint_names=Q1_POSITION_JOINTS)},
    )
    # -- interval (armed by the curriculum in stage 5)
    push_robot = EventTerm(
        func=loco_mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(10.0, 15.0),
        params={"velocity_range": {"x": (0.0, 0.0), "y": (0.0, 0.0)}},
    )


_WHEELS_J = SceneEntityCfg("robot", joint_names=Q1_WHEEL_JOINTS, preserve_order=True)
_WHEELS_B = SceneEntityCfg("contact_forces", body_names=["l_wheel_link", "r_wheel_link"], preserve_order=True)


@configclass
class RewardsCfg:
    # ---- positive task (weights are overwritten by SKATE_STAGES) ----
    wheel_speed = RewTerm(func=mdp.wheel_speed, weight=1.0, params={"command_name": "base_velocity", "std": 0.35, "asset_cfg": _WHEELS_J, "wheel_radius": WHEEL_RADIUS})
    base_vx_track = RewTerm(func=mdp.base_vx_track, weight=1.0, params={"command_name": "base_velocity", "std": 0.35})
    heading_hold = RewTerm(func=loco_mdp.track_ang_vel_z_exp, weight=0.0, params={"command_name": "base_velocity", "std": 0.5})
    leg_symmetry = RewTerm(func=mdp.leg_symmetry, weight=0.3, params={"asset_cfg": SceneEntityCfg("robot"), "pairs": Q1_LEG_PAIRS, "std": 0.5})
    grounded = RewTerm(func=mdp.grounded, weight=1.0, params={"sensor_cfg": _WHEELS_B, "command_name": "base_velocity", "threshold": 5.0})
    skating_air_time = RewTerm(
        func=mdp.skating_air_time, weight=0.0,
        params={"sensor_cfg": _WHEELS_B, "command_name": "base_velocity", "min_air": 0.05, "max_air": 0.25, "max_tilt": UPRIGHT_TILT, "min_height": MIN_STAND_Z},
    )
    forward_lean = RewTerm(
        func=mdp.forward_lean, weight=0.0,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=["torso"]), "target": 0.17, "std": 0.06, "max_tilt": UPRIGHT_TILT, "min_height": MIN_STAND_Z},
    )
    arms_back = RewTerm(
        func=mdp.arms_back, weight=0.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=_ARM_KEYFRAME_JOINTS), "std": 0.35, "max_tilt": UPRIGHT_TILT, "min_height": MIN_STAND_Z},
    )
    head_up = RewTerm(func=mdp.head_up, weight=0.2, params={"asset_cfg": SceneEntityCfg("robot", body_names=["head_link"]), "std": 0.3})
    upright = RewTerm(func=mdp.upright, weight=1.0, params={"std": 0.35})
    alive = RewTerm(func=loco_mdp.is_alive, weight=0.5)

    # ---- costs ----
    action_rate_l2 = RewTerm(func=loco_mdp.action_rate_l2, weight=ACTION_RATE_KNOTS[0][1])
    torques_ake90 = RewTerm(func=loco_mdp.joint_torques_l2, weight=-2.0e-5, params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_.*_joint", ".*_knee_joint"])})
    torques_other = RewTerm(func=loco_mdp.joint_torques_l2, weight=-1.0e-5, params={"asset_cfg": SceneEntityCfg("robot", joint_names=["waist_.*_joint", ".*_shoulder_.*_joint", ".*_elbow_joint", ".*_wheel_joint"])})
    ake90_saturation = RewTerm(func=mdp.actuator_saturation, weight=-0.2, params={"actuator_names": _AKE90_GROUPS})
    ang_vel_roll_yaw = RewTerm(func=mdp.ang_vel_roll_yaw_l2, weight=-0.05)
    lin_vel_z_l2 = RewTerm(func=loco_mdp.lin_vel_z_l2, weight=-0.5)
    undesired_contacts = RewTerm(func=loco_mdp.undesired_contacts, weight=-1.0, params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=_ILLEGAL_BODIES), "threshold": 1.0})
    roller_contact = RewTerm(func=mdp.any_contact, weight=-2.0, params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_knee_roller_link"), "threshold": 1.0})
    dof_pos_limits = RewTerm(func=loco_mdp.joint_pos_limits, weight=-1.0)
    wheel_slip = RewTerm(func=mdp.wheel_slip, weight=-0.3, params={"asset_cfg": _WHEELS_J, "sensor_cfg": _WHEELS_B, "wheel_radius": WHEEL_RADIUS})
    wheel_lateral_slip = RewTerm(func=mdp.wheel_lateral_slip, weight=-0.5, params={"asset_cfg": _WHEELS_J, "sensor_cfg": _WHEELS_B, "wheel_radius": WHEEL_RADIUS})
    flying = RewTerm(func=mdp.flying, weight=-2.0, params={"sensor_cfg": _WHEELS_B, "max_air": 0.08})
    long_air = RewTerm(func=mdp.long_air, weight=-5.0, params={"sensor_cfg": _WHEELS_B, "cap": 0.30})
    home_unused = RewTerm(func=mdp.home_pose_l1, weight=-0.1, params={"asset_cfg": SceneEntityCfg("robot", joint_names=_UNUSED_JOINTS)})
    base_height = RewTerm(func=loco_mdp.base_height_l2, weight=-1.0, params={"target_height": SKATE_PELVIS_Z})
    termination_penalty = RewTerm(func=loco_mdp.is_terminated, weight=-100.0)


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=loco_mdp.time_out, time_out=True)
    base_contact = DoneTerm(func=loco_mdp.illegal_contact, params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=["pelvis", "torso", "head_link", ".*_shoulder_.*", ".*_elbow_.*", ".*_thigh_link"]), "threshold": 1.0})
    bad_orientation = DoneTerm(func=loco_mdp.bad_orientation, params={"limit_angle": 0.6})
    base_too_low = DoneTerm(func=loco_mdp.root_height_below_minimum, params={"minimum_height": 0.60})


@configclass
class CurriculumCfg:
    stage = CurrTerm(func=mdp.skate_stage, params={"stages": SKATE_STAGES, "steps_per_iter": NUM_STEPS_PER_ENV, "start_iter": 0, "force_stage": None})
    action_rate = CurrTerm(func=mdp.action_rate_schedule, params={"knots": ACTION_RATE_KNOTS, "steps_per_iter": NUM_STEPS_PER_ENV, "start_iter": 0, "term_name": "action_rate_l2"})
    metrics = CurrTerm(
        func=mdp.skate_metrics,
        params={
            "asset_cfg": _WHEELS_J,
            "sensor_cfg": _WHEELS_B,
            "torso_cfg": SceneEntityCfg("robot", body_names=["torso"]),
            "shoulder_cfg": SceneEntityCfg("robot", joint_names=[".*_shoulder_pitch_joint"]),
            "ake90_groups": _AKE90_GROUPS,
            "wheel_radius": WHEEL_RADIUS,
        },
    )


##
# Environment
##
@configclass
class Q1SkateEnvCfg(ManagerBasedRLEnvCfg):
    scene: Q1SkateSceneCfg = Q1SkateSceneCfg(num_envs=4096, env_spacing=3.0)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()
    viewer: ViewerCfg = ViewerCfg(eye=(2.6, 2.6, 1.6), lookat=(0.0, 0.0, 0.7), origin_type="asset_root", asset_name="robot")

    def __post_init__(self):
        self.decimation = DECIMATION  # 4 -> 50 Hz policy
        self.episode_length_s = 20.0
        self.sim.dt = PHYSICS_DT  # 1/200 s = CAN rate
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15
        self.scene.contact_forces.update_period = self.sim.dt
        self.scene.imu.update_period = self.sim.dt


@configclass
class Q1SkateEnvCfg_PLAY(Q1SkateEnvCfg):
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
        self.curriculum.stage.params["force_stage"] = len(SKATE_STAGES) - 1
        self.curriculum.action_rate = None
        self.commands.base_velocity.resampling_time_range = (1.0e9, 1.0e9)
        self.commands.base_velocity.rel_standing_envs = 0.0
        self.commands.base_velocity.ranges.lin_vel_x = (-0.6, 1.8)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.6, 0.6)
