"""Native 23-actuator BU10 wheel robot, trained through Isaac Lab managers at 200/50 Hz.

This task uses the new CAD URDF unchanged in physical semantics. It deliberately has
its own policy dimensions and checkpoint experiment, because the source has no neck.
"""
from __future__ import annotations

import re
import isaaclab.sim as sim_utils
import isaaclab.envs.mdp as base_mdp
import isaaclab_tasks.manager_based.locomotion.velocity.mdp as velocity_mdp
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg, ViewerCfg
from isaaclab.managers import (
    EventTermCfg as EventTerm, ObservationGroupCfg as ObsGroup,
    ObservationTermCfg as ObsTerm, RewardTermCfg as RewTerm,
    SceneEntityCfg, TerminationTermCfg as DoneTerm,
)
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from wheel_humanoid_lab.assets.q1_inertia import (
    Q1_INERTIA_CFG, Q1_INERTIA_JOINT_ORDER, Q1_INERTIA_POSITION_JOINTS,
    Q1_INERTIA_WHEEL_JOINTS, Q1_INERTIA_MIMIC_JOINTS, Q1_INERTIA_MIMIC_DRIVERS,
    Q1_INERTIA_MIMIC_MULTIPLIERS, Q1_INERTIA_DEFAULT_POS,
    Q1_INERTIA_ROOT_Z, Q1_INERTIA_PHYSICS_DT, Q1_INERTIA_DECIMATION,
)
from . import mdp

_POSITION_SCALE = {
    "center_waist_Yaw_joint": 0.15, "center_waist_Roll_joint": 0.10, "center_waist_Pitch_joint": 0.15,
    "(left|right)_shoulder_Pitch_joint": 0.40, "(left|right)_shoulder_Roll_joint": 0.20,
    "(left|right)_shoulder_Yaw_joint": 0.20, "(left|right)_elbow_Pitch_joint": 0.30,
    "(left|right)_wrist_Yaw_joint": 0.0, "(left|right)_finger_1_Roll_joint": 0.0,
    "(left|right)_hip_Pitch_joint": 0.50, "(left|right)_hip_Roll_joint": 0.25,
    "(left|right)_knee_1_Pitch_joint": 0.45,
}
_POSITION_CLIP = {}
for _joint in Q1_INERTIA_POSITION_JOINTS:
    _scale = next(s for pattern, s in _POSITION_SCALE.items() if re.fullmatch(pattern, _joint))
    _q0 = Q1_INERTIA_DEFAULT_POS[_joint]
    _POSITION_CLIP[_joint] = (_q0 - _scale, _q0 + _scale)
# Original waist pitch has lower=0, so keep the scaled target in its physical range.
_POSITION_CLIP["center_waist_Pitch_joint"] = (0.0, 0.25)


def _joints(names):
    return SceneEntityCfg("robot", joint_names=names, preserve_order=True)


@configclass
class InertiaSceneCfg(InteractiveSceneCfg):
    terrain = TerrainImporterCfg(
        prim_path="/World/ground", terrain_type="plane", collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply", restitution_combine_mode="multiply",
            static_friction=1.0, dynamic_friction=0.9,
        ), debug_vis=False,
    )
    robot: ArticulationCfg = Q1_INERTIA_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3)
    light = AssetBaseCfg(prim_path="/World/light", spawn=sim_utils.DomeLightCfg(intensity=750.0))


@configclass
class CommandsCfg:
    base_velocity = velocity_mdp.UniformVelocityCommandCfg(
        asset_name="robot", resampling_time_range=(6.0, 10.0),
        rel_standing_envs=0.5, rel_heading_envs=0.0, heading_command=False, debug_vis=False,
        ranges=velocity_mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.3, 0.6), lin_vel_y=(0.0, 0.0), ang_vel_z=(-0.25, 0.25),
        ),
    )


@configclass
class ActionsCfg:
    joint_pos = base_mdp.JointPositionActionCfg(
        asset_name="robot", joint_names=Q1_INERTIA_POSITION_JOINTS, preserve_order=True,
        scale=_POSITION_SCALE, use_default_offset=True, clip=_POSITION_CLIP,
    )
    wheel_vel = base_mdp.JointVelocityActionCfg(
        asset_name="robot", joint_names=Q1_INERTIA_WHEEL_JOINTS, preserve_order=True,
        scale=25.0, use_default_offset=True, clip={".*": (-25.0, 25.0)},
    )
    mimic = mdp.InertiaMimicActionCfg(
        asset_name="robot", follower_joints=Q1_INERTIA_MIMIC_JOINTS,
        driver_joints=Q1_INERTIA_MIMIC_DRIVERS, multipliers=Q1_INERTIA_MIMIC_MULTIPLIERS,
    )


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        base_ang_vel = ObsTerm(func=base_mdp.base_ang_vel, scale=0.25, noise=Unoise(n_min=-0.02, n_max=0.02))
        gravity = ObsTerm(func=base_mdp.projected_gravity, noise=Unoise(n_min=-0.01, n_max=0.01))
        command = ObsTerm(func=base_mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos = ObsTerm(func=mdp.joint_pos_contract, params={"asset_cfg": _joints(Q1_INERTIA_JOINT_ORDER)})
        joint_vel = ObsTerm(func=base_mdp.joint_vel, scale=0.05, params={"asset_cfg": _joints(Q1_INERTIA_JOINT_ORDER)})
        last_action = ObsTerm(func=base_mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class CriticCfg(ObsGroup):
        base_ang_vel = ObsTerm(func=base_mdp.base_ang_vel, scale=0.25)
        gravity = ObsTerm(func=base_mdp.projected_gravity)
        command = ObsTerm(func=base_mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos = ObsTerm(func=mdp.joint_pos_contract, params={"asset_cfg": _joints(Q1_INERTIA_JOINT_ORDER)})
        joint_vel = ObsTerm(func=base_mdp.joint_vel, scale=0.05, params={"asset_cfg": _joints(Q1_INERTIA_JOINT_ORDER)})
        last_action = ObsTerm(func=base_mdp.last_action)
        base_lin_vel = ObsTerm(func=base_mdp.base_lin_vel)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class EventsCfg:
    reset_root = EventTerm(
        func=base_mdp.reset_root_state_uniform, mode="reset",
        params={"pose_range": {"x": (-0.1, 0.1), "y": (-0.1, 0.1), "yaw": (-0.1, 0.1)}, "velocity_range": {}},
    )
    reset_joints = EventTerm(
        func=base_mdp.reset_joints_by_scale, mode="reset",
        params={"position_range": (1.0, 1.0), "velocity_range": (0.0, 0.0)},
    )


@configclass
class RewardsCfg:
    alive = RewTerm(func=base_mdp.is_alive, weight=1.0)
    linear_velocity = RewTerm(func=base_mdp.track_lin_vel_xy_exp, weight=2.0,
                             params={"command_name": "base_velocity", "std": 0.5})
    yaw_velocity = RewTerm(func=base_mdp.track_ang_vel_z_exp, weight=0.5,
                          params={"command_name": "base_velocity", "std": 0.5})
    wheel_velocity = RewTerm(func=mdp.wheel_velocity_tracking, weight=0.5,
                            params={"command_name": "base_velocity", "asset_cfg": _joints(Q1_INERTIA_WHEEL_JOINTS)})
    grounded = RewTerm(func=mdp.wheels_grounded, weight=0.5,
                      params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=["left_wheel_link", "right_wheel_link"], preserve_order=True)})
    orientation = RewTerm(func=base_mdp.flat_orientation_l2, weight=-3.0)
    base_height = RewTerm(func=base_mdp.base_height_l2, weight=-3.0, params={"target_height": Q1_INERTIA_ROOT_Z})
    vertical_velocity = RewTerm(func=base_mdp.lin_vel_z_l2, weight=-0.5)
    roll_pitch_velocity = RewTerm(func=base_mdp.ang_vel_xy_l2, weight=-0.05)
    posture = RewTerm(func=base_mdp.joint_deviation_l1, weight=-0.08,
                     params={"asset_cfg": _joints(Q1_INERTIA_POSITION_JOINTS)})
    action_rate = RewTerm(func=base_mdp.action_rate_l2, weight=-0.02)
    action_magnitude = RewTerm(func=base_mdp.action_l2, weight=-0.005)
    joint_torque = RewTerm(func=base_mdp.joint_torques_l2, weight=-1.0e-5,
                          params={"asset_cfg": _joints(Q1_INERTIA_JOINT_ORDER)})
    joint_limits = RewTerm(func=base_mdp.joint_pos_limits, weight=-1.0,
                          params={"asset_cfg": _joints(Q1_INERTIA_POSITION_JOINTS)})
    mimic_error = RewTerm(func=mdp.mimic_error_l2, weight=-0.1,
                         params={"asset_cfg": _joints(Q1_INERTIA_MIMIC_JOINTS), "driver_cfg": _joints(Q1_INERTIA_MIMIC_DRIVERS)})
    termination = RewTerm(func=base_mdp.is_terminated, weight=-2.0)


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=base_mdp.time_out, time_out=True)
    fallen = DoneTerm(func=base_mdp.root_height_below_minimum, params={"minimum_height": 0.55})
    tilted = DoneTerm(func=base_mdp.bad_orientation, params={"limit_angle": 0.85})
    nonfinite = DoneTerm(func=mdp.nonfinite_state)


@configclass
class Q1InertiaSkateEnvCfg(ManagerBasedRLEnvCfg):
    scene: InertiaSceneCfg = InertiaSceneCfg(num_envs=2048, env_spacing=3.0)
    commands: CommandsCfg = CommandsCfg()
    actions: ActionsCfg = ActionsCfg()
    observations: ObservationsCfg = ObservationsCfg()
    rewards: RewardsCfg = RewardsCfg()
    events: EventsCfg = EventsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    curriculum = None

    def __post_init__(self):
        self.decimation = Q1_INERTIA_DECIMATION
        self.episode_length_s = 20.0
        self.sim.dt = Q1_INERTIA_PHYSICS_DT
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.scene.contact_forces.update_period = self.sim.dt
        self.viewer = ViewerCfg(eye=(3.0, -3.0, 2.0), lookat=(0.0, 0.0, 0.8))


@configclass
class Q1InertiaSkateEnvCfg_PLAY(Q1InertiaSkateEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.observations.policy.enable_corruption = False
