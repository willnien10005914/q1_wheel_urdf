"""Articulation configs for the dual-wheel humanoid."""

import os

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg

from wheel_humanoid_lab import WHEEL_HUMANOID_ROOT_DIR

URDF_PATH = os.path.join(WHEEL_HUMANOID_ROOT_DIR, "urdf", "wheel_humanoid.urdf")
USD_DIR = os.path.join(WHEEL_HUMANOID_ROOT_DIR, "usd")

# Standing skate pose: only the two foot wheels contact the ground.
# A small knee bend (~26 deg) is used for balance; knee rollers stay ~0.32 m up.
SKATE_PELVIS_Z = 0.875
SKATE_HIP_PITCH = -0.25
SKATE_KNEE = 0.45

WHEEL_HUMANOID_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        asset_path=URDF_PATH,
        usd_dir=USD_DIR,
        usd_file_name="wheel_humanoid.usd",
        fix_base=False,
        merge_fixed_joints=False,
        make_instanceable=True,
        activate_contact_sensors=True,
        replace_cylinders_with_capsules=False,
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0.0, damping=0.0)
        ),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=4,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.90),
        joint_pos={".*": 0.0},
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.95,
    actuators={
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[".*_hip_pitch_joint", ".*_hip_roll_joint", ".*_knee_joint"],
            effort_limit_sim=60.0,
            velocity_limit_sim=12.0,
            stiffness=120.0,
            damping=6.0,
        ),
        "wheels": ImplicitActuatorCfg(
            joint_names_expr=[".*_wheel_joint"],
            effort_limit_sim=20.0,
            velocity_limit_sim=30.0,
            stiffness=0.0,
            damping=2.0,
        ),
        "waist": ImplicitActuatorCfg(
            joint_names_expr=["waist_.*_joint"],
            effort_limit_sim=40.0,
            velocity_limit_sim=6.0,
            stiffness=60.0,
            damping=3.0,
        ),
        "arms": ImplicitActuatorCfg(
            joint_names_expr=[".*_shoulder_.*_joint", ".*_elbow_joint"],
            effort_limit_sim=30.0,
            velocity_limit_sim=10.0,
            stiffness=40.0,
            damping=2.0,
        ),
    },
)

WHEEL_HUMANOID_SKATEBOARD_CFG = WHEEL_HUMANOID_CFG.replace(
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, SKATE_PELVIS_Z),
        joint_pos={
            ".*_hip_pitch_joint": SKATE_HIP_PITCH,
            ".*_hip_roll_joint": 0.0,
            ".*_knee_joint": SKATE_KNEE,
            ".*_knee_roller_joint": 0.444444 * SKATE_KNEE,
            "waist_yaw_joint": 0.0,
            "waist_roll_joint": 0.0,
            "waist_pitch_joint": 0.08,
            "l_shoulder_pitch_joint": 0.20,
            "r_shoulder_pitch_joint": 0.20,
            "l_shoulder_roll_joint": 0.25,
            "r_shoulder_roll_joint": -0.25,
            "l_shoulder_yaw_joint": 0.0,
            "r_shoulder_yaw_joint": 0.0,
            "l_elbow_joint": -0.40,
            "r_elbow_joint": -0.40,
            ".*_wheel_joint": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
)
