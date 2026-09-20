"""Skateboard-style dual-wheel sliding environment for the wheel humanoid."""

import math

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg, ViewerCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as loco_mdp

import wheel_humanoid_lab.tasks.manager_based.skateboard.mdp as skate_mdp
from wheel_humanoid_lab.assets import WHEEL_HUMANOID_SKATEBOARD_CFG


##
# Scene
##


@configclass
class WheelHumanoidSkateboardSceneCfg(InteractiveSceneCfg):
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        terrain_generator=None,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        debug_vis=False,
    )
    robot: ArticulationCfg = WHEEL_HUMANOID_SKATEBOARD_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*",
        history_length=3,
        track_air_time=True,
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
        resampling_time_range=(8.0, 12.0),
        rel_standing_envs=0.10,
        rel_heading_envs=0.0,
        heading_command=False,
        debug_vis=True,
        ranges=loco_mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-1.5, 2.4),
            lin_vel_y=(-0.6, 0.6),
            ang_vel_z=(-2.2, 2.2),
        ),
    )


@configclass
class ActionsCfg:
    joint_pos = loco_mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[
            ".*_hip_pitch_joint",
            ".*_hip_roll_joint",
            ".*_knee_joint",
            "waist_.*_joint",
            ".*_shoulder_.*_joint",
            ".*_elbow_joint",
        ],
        scale={
            ".*_hip_pitch_joint": 0.50,
            ".*_hip_roll_joint": 0.12,
            ".*_knee_joint": 0.10,
            "waist_.*_joint": 0.20,
            ".*_shoulder_.*_joint": 0.20,
            ".*_elbow_joint": 0.20,
        },
        use_default_offset=True,
    )
    wheel_vel = loco_mdp.JointVelocityActionCfg(
        asset_name="robot",
        joint_names=[".*_wheel_joint"],
        scale=22.0,
        use_default_offset=True,
    )


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        base_lin_vel = ObsTerm(func=loco_mdp.base_lin_vel, noise=Unoise(n_min=-0.1, n_max=0.1))
        base_ang_vel = ObsTerm(func=loco_mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(func=loco_mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05))
        velocity_commands = ObsTerm(func=loco_mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos = ObsTerm(
            func=loco_mdp.joint_pos_rel,
            noise=Unoise(n_min=-0.01, n_max=0.01),
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=[
                        ".*_hip_pitch_joint",
                        ".*_hip_roll_joint",
                        ".*_knee_joint",
                        "waist_.*_joint",
                        ".*_shoulder_.*_joint",
                        ".*_elbow_joint",
                    ],
                )
            },
        )
        joint_vel = ObsTerm(
            func=loco_mdp.joint_vel_rel,
            noise=Unoise(n_min=-1.5, n_max=1.5),
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=[
                        ".*_hip_pitch_joint",
                        ".*_hip_roll_joint",
                        ".*_knee_joint",
                        "waist_.*_joint",
                        ".*_shoulder_.*_joint",
                        ".*_elbow_joint",
                        ".*_wheel_joint",
                    ],
                )
            },
        )
        actions = ObsTerm(func=loco_mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    physics_material = EventTerm(
        func=loco_mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.8, 1.2),
            "dynamic_friction_range": (0.6, 1.0),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )
    add_base_mass = EventTerm(
        func=loco_mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="pelvis"),
            "mass_distribution_params": (-1.0, 1.0),
            "operation": "add",
        },
    )
    reset_base = EventTerm(
        func=loco_mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        },
    )
    reset_robot_joints = EventTerm(
        func=loco_mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (0.95, 1.05),
            "velocity_range": (0.0, 0.0),
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    ".*_hip_pitch_joint",
                    ".*_hip_roll_joint",
                    ".*_knee_joint",
                    "waist_.*_joint",
                    ".*_shoulder_.*_joint",
                    ".*_elbow_joint",
                ],
            ),
        },
    )
    push_robot = EventTerm(
        func=loco_mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(8.0, 12.0),
        params={"velocity_range": {"x": (-0.3, 0.3), "y": (-0.3, 0.3)}},
    )


@configclass
class RewardsCfg:
    track_lin_vel_xy_exp = RewTerm(
        func=skate_mdp.lin_vel_xy_exp,
        weight=4.0,
        params={"command_name": "base_velocity", "std": math.sqrt(0.25)},
    )
    track_ang_vel_z_exp = RewTerm(
        func=loco_mdp.track_ang_vel_z_exp,
        weight=1.5,
        params={"command_name": "base_velocity", "std": math.sqrt(0.25)},
    )
    wheel_roll_match = RewTerm(
        func=skate_mdp.wheel_roll_match,
        weight=2.5,
        params={
            "command_name": "base_velocity",
            "wheel_radius": 0.10,
            "std": 4.0,
            "asset_cfg": SceneEntityCfg("robot", joint_names=["l_wheel_joint", "r_wheel_joint"]),
        },
    )
    forward_lean = RewTerm(
        func=skate_mdp.forward_lean,
        weight=0.8,
        params={"command_name": "base_velocity", "gain": 0.10, "std": 0.18},
    )
    alive = RewTerm(func=loco_mdp.is_alive, weight=0.2)
    termination_penalty = RewTerm(func=loco_mdp.is_terminated, weight=-200.0)
    lin_vel_z_l2 = RewTerm(func=loco_mdp.lin_vel_z_l2, weight=-1.0)
    ang_vel_xy_l2 = RewTerm(func=loco_mdp.ang_vel_xy_l2, weight=-0.05)
    flat_orientation_l2 = RewTerm(func=loco_mdp.flat_orientation_l2, weight=-0.8)
    dof_torques_l2 = RewTerm(
        func=loco_mdp.joint_torques_l2,
        weight=-1.0e-5,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", joint_names=[".*_hip_.*", ".*_knee_joint", "waist_.*_joint"]
            )
        },
    )
    dof_acc_l2 = RewTerm(
        func=loco_mdp.joint_acc_l2,
        weight=-2.5e-7,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_.*", ".*_knee_joint"])},
    )
    action_rate_l2 = RewTerm(func=loco_mdp.action_rate_l2, weight=-0.008)
    dof_pos_limits = RewTerm(func=loco_mdp.joint_pos_limits, weight=-1.0)
    stand_pose = RewTerm(
        func=skate_mdp.skate_pose_l2,
        weight=-0.6,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[".*_hip_pitch_joint", ".*_knee_joint"],
            )
        },
    )
    hip_roll_pose = RewTerm(
        func=skate_mdp.skate_pose_l2,
        weight=-0.8,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[".*_hip_roll_joint"],
            )
        },
    )
    arm_pose = RewTerm(
        func=skate_mdp.skate_pose_l2,
        weight=-0.15,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=["waist_.*_joint", ".*_shoulder_.*", ".*_elbow_joint"],
            )
        },
    )
    base_height = RewTerm(
        func=loco_mdp.base_height_l2,
        weight=-3.0,
        params={"target_height": 0.87},
    )
    wheel_contact = RewTerm(
        func=skate_mdp.both_contacts,
        weight=0.8,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_wheel_link"), "threshold": 1.0},
    )
    roller_contact = RewTerm(
        func=skate_mdp.both_contacts,
        weight=-4.0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_knee_roller_link"), "threshold": 1.0},
    )
    undesired_contacts = RewTerm(
        func=loco_mdp.undesired_contacts,
        weight=-2.0,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=["pelvis", "torso", ".*_shoulder_.*", ".*_elbow_.*", "waist_.*", ".*_thigh_link"],
            ),
            "threshold": 1.0,
        },
    )


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=loco_mdp.time_out, time_out=True)
    base_contact = DoneTerm(
        func=loco_mdp.illegal_contact,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=["pelvis", "torso", ".*_shoulder_.*", ".*_elbow_.*"],
            ),
            "threshold": 1.0,
        },
    )
    bad_orientation = DoneTerm(
        func=loco_mdp.bad_orientation,
        params={"limit_angle": 1.0},
    )
    base_too_low = DoneTerm(
        func=loco_mdp.root_height_below_minimum,
        params={"minimum_height": 0.55},
    )


@configclass
class WheelHumanoidSkateboardEnvCfg(ManagerBasedRLEnvCfg):
    scene: WheelHumanoidSkateboardSceneCfg = WheelHumanoidSkateboardSceneCfg(num_envs=1024, env_spacing=3.0)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    viewer: ViewerCfg = ViewerCfg(eye=(3.0, 3.0, 1.8), lookat=(0.0, 0.0, 0.7), origin_type="asset_root", asset_name="robot")

    def __post_init__(self):
        self.decimation = 4
        self.episode_length_s = 20.0
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15
        if self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = self.sim.dt


@configclass
class WheelHumanoidSkateboardEnvCfg_PLAY(WheelHumanoidSkateboardEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.scene.env_spacing = 2.5
        self.episode_length_s = 120.0
        self.observations.policy.enable_corruption = False
        self.events.push_robot = None
        self.events.add_base_mass = None
        self.events.reset_base.params["pose_range"] = {"x": (0.0, 0.0), "y": (0.0, 0.0), "yaw": (0.0, 0.0)}
        self.commands.base_velocity.resampling_time_range = (1.0e9, 1.0e9)
        self.commands.base_velocity.rel_standing_envs = 0.0
        self.commands.base_velocity.heading_command = False
        self.commands.base_velocity.debug_vis = True
        self.commands.base_velocity.ranges.lin_vel_x = (-1.8, 2.4)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.8, 0.8)
        self.commands.base_velocity.ranges.ang_vel_z = (-2.2, 2.2)
