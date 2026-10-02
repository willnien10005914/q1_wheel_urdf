"""Base Q1 floor-recovery env (shared by recovery_v3).

Actor obs = skate proprio (78) + recovery command block (40) = 118.
Critic adds privileged base/wheel terms to reach 128.
"""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg, ViewerCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, ImuCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveGaussianNoiseCfg as Gnoise

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as loco_mdp

import wheel_humanoid_lab.tasks.manager_based.skate.mdp as skate_mdp
from wheel_humanoid_lab.assets import (
    DECIMATION,
    IMU_CFG,
    PHYSICS_DT,
    Q1_JOINT_ORDER,
    Q1_POSITION_JOINTS,
    Q1_WHEEL_CUBEMARS_CFG,
    Q1_WHEEL_JOINTS,
)
from wheel_humanoid_lab.tasks.manager_based.skate.skate_env_cfg import EventCfg


@configclass
class Q1RecoverySceneCfg(InteractiveSceneCfg):
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


def _zero_command(env):
    """Placeholder; recovery_v3 overrides with mdp.observation (40-D)."""
    import torch

    return torch.zeros((env.num_envs, 40), device=env.device)


@configclass
class RecoveryActionsCfg:
    """Slots filled by recovery_v3 Actions (contact-gated position + wheel vel)."""

    joint_pos = None
    wheel_vel = None


@configclass
class RecoveryObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        imu_gyro = ObsTerm(
            func=skate_mdp.ImuObs,
            params={
                "sensor_cfg": SceneEntityCfg("imu"),
                "quantity": "ang_vel",
                "max_delay": IMU_CFG["sim"]["delay_steps"][1] // 2 + 1,
                "tilt_deg": IMU_CFG["sim"]["tilt_dr_deg"],
            },
            noise=Gnoise(std=IMU_CFG["sim"]["gyro_noise_std_rad_s"]),
        )
        imu_projected_gravity = ObsTerm(
            func=skate_mdp.ImuObs,
            params={
                "sensor_cfg": SceneEntityCfg("imu"),
                "quantity": "projected_gravity",
                "max_delay": IMU_CFG["sim"]["delay_steps"][1] // 2 + 1,
                "tilt_deg": IMU_CFG["sim"]["tilt_dr_deg"],
            },
            noise=Gnoise(std=IMU_CFG["sim"]["grav_noise_std"]),
        )
        joint_pos = ObsTerm(
            func=skate_mdp.JointPosContract,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=Q1_JOINT_ORDER, preserve_order=True), "bias_deg": 1.5},
            noise=Gnoise(std=0.005),
        )
        joint_vel = ObsTerm(
            func=skate_mdp.JointVelContract,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=Q1_JOINT_ORDER, preserve_order=True), "delay": 1},
            noise=Gnoise(std=0.3),
        )
        last_action = ObsTerm(func=loco_mdp.last_action)
        command = ObsTerm(func=_zero_command)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class CriticCfg(ObsGroup):
        base_lin_vel = ObsTerm(func=loco_mdp.base_lin_vel)
        base_height = ObsTerm(func=loco_mdp.base_pos_z)
        joint_pos = ObsTerm(
            func=skate_mdp.joint_pos_contract_clean,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=Q1_JOINT_ORDER, preserve_order=True)},
        )
        joint_vel = ObsTerm(
            func=loco_mdp.joint_vel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=Q1_JOINT_ORDER, preserve_order=True)},
        )
        last_action = ObsTerm(func=loco_mdp.last_action)
        command = ObsTerm(func=_zero_command)
        wheel_rim_speed = ObsTerm(
            func=skate_mdp.wheel_rim_speed,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=Q1_WHEEL_JOINTS, preserve_order=True)},
        )
        wheel_force = ObsTerm(
            func=skate_mdp.wheel_contact_normal_force,
            params={
                "sensor_cfg": SceneEntityCfg(
                    "contact_forces", body_names=["l_wheel_link", "r_wheel_link"], preserve_order=True
                )
            },
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class Q1RecoveryEnvCfg(ManagerBasedRLEnvCfg):
    """Base floor→kneel→stand recovery env. Rewards/actions come from recovery_v3."""

    scene: Q1RecoverySceneCfg = Q1RecoverySceneCfg(num_envs=4096, env_spacing=5.0)
    observations: RecoveryObservationsCfg = RecoveryObservationsCfg()
    actions: RecoveryActionsCfg = RecoveryActionsCfg()
    commands = None
    events: EventCfg = EventCfg()
    viewer: ViewerCfg = ViewerCfg(
        eye=(2.8, -3.2, 2.0), lookat=(0.0, 0.0, 0.5), origin_type="env", env_index=0
    )

    def __post_init__(self):
        self.decimation = DECIMATION
        self.episode_length_s = 40.0
        self.sim.dt = PHYSICS_DT
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15
        self.scene.contact_forces.update_period = self.sim.dt
        self.scene.imu.update_period = self.sim.dt
        # recovery_v3.reset writes root/joints; disable skate pose resets that would fight it.
        self.events.reset_base = None
        self.events.reset_robot_joints = None
        self.events.push_robot = None
